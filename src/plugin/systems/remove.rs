use crate::dynamics::ImpulseJoint;
use crate::dynamics::MultibodyJoint;
use crate::dynamics::RapierImpulseJointHandle;
use crate::dynamics::RapierMultibodyJointHandle;
use crate::dynamics::RapierRigidBodyHandle;
use crate::dynamics::RigidBody;
use crate::geometry::Collider;
use crate::geometry::ColliderDisabled;
use crate::geometry::RapierColliderHandle;
use crate::plugin::context::{
    RapierContextColliders, RapierContextJoints, RapierContextSimulation, RapierRigidBodySet,
};
use crate::prelude::MassModifiedEvent;
use crate::prelude::RigidBodyDisabled;
use crate::prelude::Sensor;
use bevy::ecs::query::IterQueryData;
use bevy::prelude::*;

/// System responsible for removing from Rapier the rigid-bodies/colliders/joints which had
/// their related `bevy_rapier` components removed by the user (through component removal or
/// despawn).
pub fn sync_removals(
    mut commands: Commands,
    mut context_writer: Query<(
        &mut RapierContextSimulation,
        &mut RapierContextColliders,
        &mut RapierContextJoints,
        &mut RapierRigidBodySet,
    )>,
    // Sometimes a Remove immediately followed by Add happens. These `q_has_*` queries prevent that immediate Add
    // from being removed by this system by verifying it's still removed.
    (
        q_has_rigidbody_handle,
        q_has_collider_handle,
        q_has_multibody_joint_handle,
        q_has_impulse_joint_handle,
    ): (
        Query<(), With<RapierRigidBodyHandle>>,
        Query<(), With<RapierColliderHandle>>,
        Query<(), With<RapierMultibodyJointHandle>>,
        Query<(), With<RapierImpulseJointHandle>>,
    ),
    mut removed_bodies: RemovedComponents<RapierRigidBodyHandle>,
    mut removed_colliders: RemovedComponents<RapierColliderHandle>,
    mut removed_impulse_joints: RemovedComponents<RapierImpulseJointHandle>,
    mut removed_multibody_joints: RemovedComponents<RapierMultibodyJointHandle>,
    orphan_bodies: Query<Entity, (With<RapierRigidBodyHandle>, Without<RigidBody>)>,
    orphan_colliders: Query<Entity, (With<RapierColliderHandle>, Without<Collider>)>,
    orphan_impulse_joints: Query<Entity, (With<RapierImpulseJointHandle>, Without<ImpulseJoint>)>,
    orphan_multibody_joints: Query<
        Entity,
        (With<RapierMultibodyJointHandle>, Without<MultibodyJoint>),
    >,

    mut removed_sensors: RemovedComponents<Sensor>,
    mut removed_rigid_body_disabled: RemovedComponents<RigidBodyDisabled>,
    mut removed_colliders_disabled: RemovedComponents<ColliderDisabled>,

    mut mass_modified: MessageWriter<MassModifiedEvent>,
) {
    /*
     * Rigid-bodies removal detection.
     */
    for entity in removed_bodies
        .read()
        .filter(|e| !q_has_rigidbody_handle.contains(*e))
    {
        let Some(((mut context, mut context_colliders, mut joints, mut rigidbody_set), handle)) =
            find_context(&mut context_writer, |res| res.3.entity2body.remove(&entity))
        else {
            continue;
        };
        let context = &mut *context;
        let joints = &mut *joints;

        // Removed generations have no ECS destination and must not accumulate while paused.
        rigidbody_set.discard_body_from_writeback(handle);
        let _ = rigidbody_set.last_body_transform_set.remove(&handle);
        rigidbody_set.bodies.remove(
            handle,
            &mut context.islands,
            &mut context_colliders.colliders,
            &mut joints.impulse_joints,
            &mut joints.multibody_joints,
            false,
        );
    }

    for entity in orphan_bodies.iter() {
        if let Some(((mut context, mut context_colliders, mut joints, mut rigidbody_set), handle)) =
            find_context(&mut context_writer, |res| res.3.entity2body.remove(&entity))
        {
            let context = &mut *context;
            let joints = &mut *joints;
            // Removed generations have no ECS destination and must not accumulate while paused.
            rigidbody_set.discard_body_from_writeback(handle);
            let _ = rigidbody_set.last_body_transform_set.remove(&handle);
            rigidbody_set.bodies.remove(
                handle,
                &mut context.islands,
                &mut context_colliders.colliders,
                &mut joints.impulse_joints,
                &mut joints.multibody_joints,
                false,
            );
        }
        commands.entity(entity).remove::<RapierRigidBodyHandle>();
    }

    /*
     * Collider removal detection.
     */
    for entity in removed_colliders
        .read()
        .filter(|e| !q_has_collider_handle.contains(*e))
    {
        let Some(((mut context, mut context_colliders, _, mut rigidbody_set), handle)) =
            find_context(&mut context_writer, |res| {
                res.1.entity2collider.remove(&entity)
            })
        else {
            continue;
        };
        let context = &mut *context;

        // Resolve the backend parent before removing the collider-to-entity mapping.
        let parent_handle = context_colliders
            .colliders
            .get(handle)
            .and_then(|collider| collider.parent());
        if let Some(parent_handle) = parent_handle {
            if let Some(parent) = rigidbody_set.rigid_body_entity(parent_handle) {
                mass_modified.write(parent.into());
            }
            rigidbody_set.queue_body_for_writeback(parent_handle);
        }

        context_colliders.colliders.remove(
            handle,
            &mut context.islands,
            &mut rigidbody_set.bodies,
            true,
        );
        context.deleted_colliders.insert(handle, entity);
    }

    for entity in orphan_colliders.iter() {
        if let Some(((mut context, mut context_colliders, _, mut rigidbody_set), handle)) =
            find_context(&mut context_writer, |res| {
                res.1.entity2collider.remove(&entity)
            })
        {
            let context = &mut *context;
            let context_colliders = &mut *context_colliders;

            // Resolve the backend parent before removing the orphaned collider mapping.
            let parent_handle = context_colliders
                .colliders
                .get(handle)
                .and_then(|collider| collider.parent());
            if let Some(parent_handle) = parent_handle {
                if let Some(parent) = rigidbody_set.rigid_body_entity(parent_handle) {
                    mass_modified.write(parent.into());
                }
                rigidbody_set.queue_body_for_writeback(parent_handle);
            }

            context_colliders.colliders.remove(
                handle,
                &mut context.islands,
                &mut rigidbody_set.bodies,
                true,
            );
            context.deleted_colliders.insert(handle, entity);
        }
        commands.entity(entity).remove::<RapierColliderHandle>();
    }

    /*
     * Impulse joint removal detection.
     */
    for entity in removed_impulse_joints
        .read()
        .filter(|e| !q_has_impulse_joint_handle.contains(*e))
    {
        let Some(((_, _, mut joints, mut rigidbody_set), handle)) =
            find_context(&mut context_writer, |res| {
                res.2.entity2impulse_joint.remove(&entity)
            })
        else {
            continue;
        };
        if let Some(joint) = joints.impulse_joints.remove(handle, true) {
            // Joint removal can wake either endpoint outside the scheduled step.
            rigidbody_set.queue_body_for_writeback(joint.body1());
            rigidbody_set.queue_body_for_writeback(joint.body2());
        }
    }

    for entity in orphan_impulse_joints.iter() {
        if let Some(((_, _, mut joints, mut rigidbody_set), handle)) =
            find_context(&mut context_writer, |res| {
                res.2.entity2impulse_joint.remove(&entity)
            })
        {
            if let Some(joint) = joints.impulse_joints.remove(handle, true) {
                // Joint removal can wake either endpoint outside the scheduled step.
                rigidbody_set.queue_body_for_writeback(joint.body1());
                rigidbody_set.queue_body_for_writeback(joint.body2());
            }
        }
        commands.entity(entity).remove::<RapierImpulseJointHandle>();
    }

    /*
     * Multibody joint removal detection.
     */
    for entity in removed_multibody_joints
        .read()
        .filter(|e| !q_has_multibody_joint_handle.contains(*e))
    {
        let Some(((_, _, mut joints, mut rigidbody_set), handle)) =
            find_context(&mut context_writer, |res| {
                res.2.entity2multibody_joint.remove(&entity)
            })
        else {
            continue;
        };

        // Snapshot both links because removal consumes the multibody joint record.
        let endpoints = joints
            .multibody_joints
            .get(handle)
            .and_then(|(multibody, link_id)| {
                let link = multibody.link(link_id)?;
                let parent = link
                    .parent_id()
                    .and_then(|parent_id| multibody.link(parent_id))?;
                Some((parent.rigid_body_handle(), link.rigid_body_handle()))
            });
        joints.multibody_joints.remove(handle, true);

        // Multibody removal can wake either endpoint outside the scheduled step.
        if let Some((body1, body2)) = endpoints {
            rigidbody_set.queue_body_for_writeback(body1);
            rigidbody_set.queue_body_for_writeback(body2);
        }
    }

    for entity in orphan_multibody_joints.iter() {
        if let Some(((_, _, mut joints, mut rigidbody_set), handle)) =
            find_context(&mut context_writer, |res| {
                res.2.entity2multibody_joint.remove(&entity)
            })
        {
            // Snapshot both orphaned links because removal consumes the joint record.
            let endpoints = joints
                .multibody_joints
                .get(handle)
                .and_then(|(multibody, link_id)| {
                    let link = multibody.link(link_id)?;
                    let parent = link
                        .parent_id()
                        .and_then(|parent_id| multibody.link(parent_id))?;
                    Some((parent.rigid_body_handle(), link.rigid_body_handle()))
                });
            joints.multibody_joints.remove(handle, true);

            // Multibody removal can wake either endpoint outside the scheduled step.
            if let Some((body1, body2)) = endpoints {
                rigidbody_set.queue_body_for_writeback(body1);
                rigidbody_set.queue_body_for_writeback(body2);
            }
        }
        commands
            .entity(entity)
            .remove::<RapierMultibodyJointHandle>();
    }

    /*
     * Marker components removal detection.
     */
    for entity in removed_sensors.read() {
        if let Some((mut context, handle)) = find_context(&mut context_writer, |context| {
            context.1.entity2collider.get(&entity).copied()
        }) {
            if let Some(co) = context.1.colliders.get_mut(handle) {
                co.set_sensor(false);
            }
        }
    }

    for entity in removed_colliders_disabled.read() {
        if let Some((mut context, handle)) = find_context(&mut context_writer, |context| {
            context.1.entity2collider.get(&entity).copied()
        }) {
            // Snapshot the parent before mutably re-enabling the collider.
            let parent_handle = context
                .1
                .colliders
                .get(handle)
                .and_then(|collider| collider.parent());
            if let Some(co) = context.1.colliders.get_mut(handle) {
                co.set_enabled(true);
            }

            // Re-enabling an attached collider can wake or renormalize its parent.
            if let Some(parent_handle) = parent_handle {
                context.3.queue_body_for_writeback(parent_handle);
            }
        }
    }

    for entity in removed_rigid_body_disabled.read() {
        if let Some(((_, _, _, mut rigidbody_set), handle)) =
            find_context(&mut context_writer, |res| {
                res.3.entity2body.get(&entity).copied()
            })
        {
            if let Some(rb) = rigidbody_set.bodies.get_mut(handle) {
                rb.set_enabled(true);

                // Marker removal is an explicit backend transition even without a substep.
                rigidbody_set.queue_body_for_writeback(handle);
            }
        }
    }

    // TODO: what about removing forces?
}

fn find_context<'a, TReturn, TQueryParams: IterQueryData>(
    context_writer: &'a mut Query<TQueryParams>,
    item_finder: impl Fn(&mut TQueryParams::Item<'_, '_>) -> Option<TReturn>,
) -> Option<(TQueryParams::Item<'a, 'a>, TReturn)> {
    let ret: Option<(TQueryParams::Item<'_, '_>, TReturn)> = context_writer
        .iter_mut()
        .find_map(|mut context| item_finder(&mut context).map(|handle| (context, handle)));
    ret
}
