use crate::dynamics::ImpulseJoint;
use crate::dynamics::MultibodyJoint;
use crate::dynamics::RapierImpulseJointHandle;
use crate::dynamics::RapierMultibodyJointHandle;
use crate::plugin::context::systemparams::RAPIER_CONTEXT_EXPECT_ERROR;
use crate::plugin::context::DefaultRapierContext;
use crate::plugin::context::RapierContextEntityLink;
use crate::plugin::context::RapierContextJoints;
use crate::plugin::context::RapierRigidBodySet;
use bevy::prelude::*;

/// System responsible for creating new Rapier joints from the related `bevy_rapier` components.
pub fn init_joints(
    mut commands: Commands,
    mut context_access: Query<(&mut RapierRigidBodySet, &mut RapierContextJoints)>,
    default_context_access: Query<Entity, With<DefaultRapierContext>>,
    impulse_joints: Query<
        (Entity, Option<&RapierContextEntityLink>, &ImpulseJoint),
        Without<RapierImpulseJointHandle>,
    >,
    multibody_joints: Query<
        (Entity, Option<&RapierContextEntityLink>, &MultibodyJoint),
        Without<RapierMultibodyJointHandle>,
    >,
    child_of_query: Query<&ChildOf>,
) {
    for (entity, entity_context_link, joint) in impulse_joints.iter() {
        // Get rapier context from RapierContextEntityLink or insert its default value.
        let context_entity = entity_context_link.map_or_else(
            || {
                let context_entity = default_context_access.single().ok()?;
                commands
                    .entity(entity)
                    .insert(RapierContextEntityLink(context_entity));
                Some(context_entity)
            },
            |link| Some(link.0),
        );
        let Some(context_entity) = context_entity else {
            continue;
        };

        let Ok(rigidbody_set_joints) = context_access.get_mut(context_entity) else {
            log::error!("Could not find entity {context_entity} with rapier context while initializing {entity}");
            continue;
        };
        let mut rigidbody_set = rigidbody_set_joints.0;
        let mut target = None;
        let mut body_entity = entity;
        while target.is_none() {
            target = rigidbody_set.entity2body.get(&body_entity).copied();
            if let Ok(child_of) = child_of_query.get(body_entity) {
                body_entity = child_of.parent();
            } else {
                break;
            }
        }
        let joints = rigidbody_set_joints.1.into_inner();

        if let (Some(target), Some(source)) = (
            target,
            rigidbody_set.entity2body.get(&joint.parent).copied(),
        ) {
            let handle = joints.impulse_joints.insert(
                source,
                target,
                joint.data.as_ref().into_rapier(),
                true,
            );
            commands
                .entity(entity)
                .insert(RapierImpulseJointHandle(handle));
            joints.entity2impulse_joint.insert(entity, handle);

            // Joint insertion can wake either endpoint outside the scheduled step.
            rigidbody_set.queue_body_for_writeback(source);
            rigidbody_set.queue_body_for_writeback(target);
        }
    }

    for (entity, entity_context_link, joint) in multibody_joints.iter() {
        // Get rapier context from RapierContextEntityLink or insert its default value.
        let context_entity = entity_context_link.map_or_else(
            || {
                let context_entity = default_context_access.single().ok()?;
                commands
                    .entity(entity)
                    .insert(RapierContextEntityLink(context_entity));
                Some(context_entity)
            },
            |link| Some(link.0),
        );
        let Some(context_entity) = context_entity else {
            continue;
        };

        let Ok(context_joints) = context_access.get_mut(context_entity) else {
            log::error!("Could not find entity {context_entity} with rapier context while initializing {entity}");
            continue;
        };
        let mut context = context_joints.0;
        let target = context.entity2body.get(&entity).copied();
        let joints = context_joints.1.into_inner();

        if let (Some(target), Some(source)) =
            (target, context.entity2body.get(&joint.parent).copied())
        {
            if let Some(handle) = joints.multibody_joints.insert(
                source,
                target,
                joint.data.as_ref().into_rapier(),
                true,
            ) {
                commands
                    .entity(entity)
                    .insert(RapierMultibodyJointHandle(handle));
                joints.entity2multibody_joint.insert(entity, handle);

                // Multibody insertion can wake either endpoint outside the scheduled step.
                context.queue_body_for_writeback(source);
                context.queue_body_for_writeback(target);
            } else {
                log::error!("Failed to create multibody joint: loop detected.")
            }
        }
    }
}

/// System responsible for applying changes the user made to a joint component.
pub fn apply_joint_user_changes(
    mut context: Query<(&mut RapierRigidBodySet, &mut RapierContextJoints)>,
    changed_impulse_joints: Query<
        (
            &RapierContextEntityLink,
            &RapierImpulseJointHandle,
            &ImpulseJoint,
        ),
        Changed<ImpulseJoint>,
    >,
    changed_multibody_joints: Query<
        (
            &RapierContextEntityLink,
            &RapierMultibodyJointHandle,
            &MultibodyJoint,
        ),
        Changed<MultibodyJoint>,
    >,
) {
    // TODO: right now, we only support propagating changes made to the joint data.
    //       Re-parenting the joint isn’t supported yet.
    for (link, handle, changed_joint) in changed_impulse_joints.iter() {
        let (mut rigidbody_set, mut joints) =
            context.get_mut(link.0).expect(RAPIER_CONTEXT_EXPECT_ERROR);

        // Snapshot endpoints before the mutable joint update invalidates the immutable borrow.
        let endpoints = joints
            .impulse_joints
            .get(handle.0)
            .map(|joint| (joint.body1(), joint.body2()));
        if let Some(joint) = joints.impulse_joints.get_mut(handle.0, false) {
            joint.data = changed_joint.data.as_ref().into_rapier();
        }

        // Changed constraints can wake either exact endpoint before a simulation substep.
        if let Some((body1, body2)) = endpoints {
            rigidbody_set.queue_body_for_writeback(body1);
            rigidbody_set.queue_body_for_writeback(body2);
        }
    }

    for (link, handle, changed_joint) in changed_multibody_joints.iter() {
        let (mut rigidbody_set, mut joints) =
            context.get_mut(link.0).expect(RAPIER_CONTEXT_EXPECT_ERROR);

        // Resolve both multibody links before mutating their joint description.
        let endpoints = joints
            .multibody_joints
            .get(handle.0)
            .and_then(|(multibody, link_id)| {
                let link = multibody.link(link_id)?;
                let parent = link
                    .parent_id()
                    .and_then(|parent_id| multibody.link(parent_id))?;
                Some((parent.rigid_body_handle(), link.rigid_body_handle()))
            });
        // TODO: not sure this will always work properly, e.g., if the number of Dofs is changed.
        if let Some((mb, link_id)) = joints.multibody_joints.get_mut(handle.0) {
            if let Some(link) = mb.link_mut(link_id) {
                link.joint.data = changed_joint.data.as_ref().into_rapier();
            }
        }

        // Changed multibody constraints can wake either exact endpoint before a substep.
        if let Some((body1, body2)) = endpoints {
            rigidbody_set.queue_body_for_writeback(body1);
            rigidbody_set.queue_body_for_writeback(body2);
        }
    }
}
