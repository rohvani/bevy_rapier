use crate::dynamics::RapierRigidBodyHandle;
use crate::plugin::context::systemparams::RAPIER_CONTEXT_EXPECT_ERROR;
use crate::plugin::context::{
    DefaultRapierContext, RapierContextColliders, RapierContextEntityLink, RapierRigidBodySet,
};
use crate::plugin::{configuration::TimestepMode, RapierConfiguration};
use crate::{dynamics::RigidBody, plugin::context::SimulationToRenderTime};
use crate::{prelude::*, utils};
use bevy::prelude::*;
use rapier::dynamics::{RigidBodyBuilder, RigidBodyHandle, RigidBodyType};
use std::collections::HashMap;

/// Components that will be updated after a physics step.
pub type RigidBodyWritebackComponents<'a> = (
    &'a RapierRigidBodyHandle,
    &'a RapierContextEntityLink,
    Option<&'a ChildOf>,
    Option<&'a mut Transform>,
    Option<&'a mut TransformInterpolation>,
    Option<&'a mut Velocity>,
    Option<&'a mut Sleeping>,
);

/// Components related to rigid-bodies.
pub type RigidBodyComponents<'a> = (
    (Entity, Option<&'a RapierContextEntityLink>),
    &'a RigidBody,
    Option<&'a GlobalTransform>,
    Option<&'a Velocity>,
    Option<&'a AdditionalMassProperties>,
    Option<&'a ReadMassProperties>,
    Option<&'a LockedAxes>,
    Option<&'a ExternalForce>,
    Option<&'a GravityScale>,
    (Option<&'a Ccd>, Option<&'a SoftCcd>),
    Option<&'a Dominance>,
    Option<&'a Sleeping>,
    Option<&'a Damping>,
    Option<&'a RigidBodyDisabled>,
    Option<&'a AdditionalSolverIterations>,
);

/// System responsible for applying changes the user made to a rigid-body-related component.
pub fn apply_rigid_body_user_changes(
    mut rigid_body_sets: Query<&mut RapierRigidBodySet>,
    config: Query<&RapierConfiguration>,
    changed_rb_types: Query<
        (&RapierRigidBodyHandle, &RapierContextEntityLink, &RigidBody),
        Changed<RigidBody>,
    >,
    mut changed_transforms: Query<
        (
            &RapierRigidBodyHandle,
            &RapierContextEntityLink,
            &GlobalTransform,
            Option<&mut TransformInterpolation>,
        ),
        Changed<GlobalTransform>,
    >,
    changed_velocities: Query<
        (&RapierRigidBodyHandle, &RapierContextEntityLink, &Velocity),
        Changed<Velocity>,
    >,
    changed_additional_mass_props: Query<
        (
            Entity,
            &RapierContextEntityLink,
            &RapierRigidBodyHandle,
            &AdditionalMassProperties,
        ),
        Changed<AdditionalMassProperties>,
    >,
    changed_locked_axes: Query<
        (
            &RapierRigidBodyHandle,
            &RapierContextEntityLink,
            &LockedAxes,
        ),
        Changed<LockedAxes>,
    >,
    changed_forces: Query<
        (
            &RapierRigidBodyHandle,
            &RapierContextEntityLink,
            &ExternalForce,
        ),
        Changed<ExternalForce>,
    >,
    mut changed_impulses: Query<
        (
            &RapierRigidBodyHandle,
            &RapierContextEntityLink,
            &mut ExternalImpulse,
        ),
        Changed<ExternalImpulse>,
    >,
    changed_gravity_scale: Query<
        (
            &RapierRigidBodyHandle,
            &RapierContextEntityLink,
            &GravityScale,
        ),
        Changed<GravityScale>,
    >,
    (changed_ccd, changed_soft_ccd): (
        Query<(&RapierRigidBodyHandle, &RapierContextEntityLink, &Ccd), Changed<Ccd>>,
        Query<(&RapierRigidBodyHandle, &RapierContextEntityLink, &SoftCcd), Changed<SoftCcd>>,
    ),
    changed_dominance: Query<
        (&RapierRigidBodyHandle, &RapierContextEntityLink, &Dominance),
        Changed<Dominance>,
    >,
    changed_sleeping: Query<
        (&RapierRigidBodyHandle, &RapierContextEntityLink, &Sleeping),
        Changed<Sleeping>,
    >,
    changed_damping: Query<
        (&RapierRigidBodyHandle, &RapierContextEntityLink, &Damping),
        Changed<Damping>,
    >,
    (changed_disabled, changed_additional_solver_iterations): (
        Query<
            (
                &RapierRigidBodyHandle,
                &RapierContextEntityLink,
                &RigidBodyDisabled,
            ),
            Changed<RigidBodyDisabled>,
        >,
        Query<
            (
                &RapierRigidBodyHandle,
                &RapierContextEntityLink,
                &AdditionalSolverIterations,
            ),
            Changed<AdditionalSolverIterations>,
        >,
    ),
    mut mass_modified: MessageWriter<MassModifiedEvent>,
) {
    // Deal with sleeping first, because other changes may then wake-up the
    // rigid-body again.
    for (handle, link, sleeping) in changed_sleeping.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();

        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            let activation = rb.activation_mut();
            activation.normalized_linear_threshold = sleeping.normalized_linear_threshold;
            activation.angular_threshold = sleeping.angular_threshold;

            if !sleeping.sleeping && activation.sleeping {
                rb.wake_up(true);
            } else if sleeping.sleeping && !activation.sleeping {
                rb.sleep();
            }

            // Publish explicit wake/sleep transitions even if no simulation substep follows.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    // NOTE: we must change the rigid-body type before updating the
    //       transform or velocity. Otherwise, if the rigid-body was fixed
    //       and changed to anything else, the velocity change wouldn’t have any effect.
    //       Similarly, if the rigid-body was kinematic position-based before and
    //       changed to anything else, a transform change would modify the next
    //       position instead of the current one.
    for (handle, link, rb_type) in changed_rb_types.iter() {
        let context = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = context.bodies.get_mut(handle.0) {
            rb.set_body_type((*rb_type).into(), true);

            // Body-type changes can normalize velocity or activation outside an island step.
            context.queue_body_for_writeback(handle.0);
        }
    }

    // Manually checks if the transform changed.
    // This is needed for detecting if the user actually changed the rigid-body
    // transform, or if it was just the change we made in our `writeback_rigid_bodies`
    // system.
    let transform_changed_fn =
        |handle: &RigidBodyHandle,
         config: &RapierConfiguration,
         transform: &GlobalTransform,
         last_transform_set: &HashMap<RigidBodyHandle, GlobalTransform>| {
            if config.force_update_from_transform_changes {
                true
            } else if let Some(prev) = last_transform_set.get(handle) {
                *prev != *transform
            } else {
                true
            }
        };

    for (handle, link, global_transform, mut interpolation) in changed_transforms.iter_mut() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        let config = config
            .get(link.0)
            .expect("Could not get `RapierConfiguration`");
        // Use an Option<bool> to avoid running the check twice.
        let mut transform_changed = None;

        if let Some(interpolation) = interpolation.as_deref_mut() {
            transform_changed = transform_changed.or_else(|| {
                Some(transform_changed_fn(
                    &handle.0,
                    config,
                    global_transform,
                    &rigidbody_set.last_body_transform_set,
                ))
            });

            if transform_changed == Some(true) {
                // Reset the interpolation so we don’t overwrite
                // the user’s input.
                interpolation.start = None;
                interpolation.end = None;
            }
        }
        // TODO: avoid to run multiple times the mutable deref ?
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            transform_changed = transform_changed.or_else(|| {
                Some(transform_changed_fn(
                    &handle.0,
                    config,
                    global_transform,
                    &rigidbody_set.last_body_transform_set,
                ))
            });

            match rb.body_type() {
                RigidBodyType::KinematicPositionBased => {
                    if transform_changed == Some(true) {
                        rb.set_next_kinematic_position(utils::transform_to_iso(
                            &global_transform.compute_transform(),
                        ));
                        rigidbody_set
                            .last_body_transform_set
                            .insert(handle.0, *global_transform);

                        // Position-based kinematics publish only their current accepted pose.
                        rigidbody_set.queue_body_for_writeback(handle.0);
                    }
                }
                _ => {
                    if transform_changed == Some(true) {
                        rb.set_position(
                            utils::transform_to_iso(&global_transform.compute_transform()),
                            true,
                        );
                        rigidbody_set
                            .last_body_transform_set
                            .insert(handle.0, *global_transform);

                        // Direct pose changes can wake or normalize the backend immediately.
                        rigidbody_set.queue_body_for_writeback(handle.0);
                    }
                }
            }
        }
    }

    for (handle, link, velocity) in changed_velocities.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            rb.set_linvel(velocity.linear, true);
            #[allow(clippy::useless_conversion)] // Need to convert if dim3 enabled
            rb.set_angvel(velocity.angular.into(), true);

            // Rapier may ignore or normalize velocity for the current body type.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    for (entity, link, handle, mprops) in changed_additional_mass_props.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            match mprops {
                AdditionalMassProperties::MassProperties(mprops) => {
                    rb.set_additional_mass_properties(mprops.into_rapier(), true);
                }
                AdditionalMassProperties::Mass(mass) => {
                    rb.set_additional_mass(*mass, true);
                }
            }

            mass_modified.write(entity.into());

            // Mass edits may wake the body or change the velocity Rapier accepts.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    for (handle, link, additional_solver_iters) in changed_additional_solver_iterations.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            rb.set_additional_solver_iterations(additional_solver_iters.0);

            // Retain the body in case Rapier adjusts activation for the explicit edit.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    for (handle, link, locked_axes) in changed_locked_axes.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            rb.set_locked_axes((*locked_axes).into(), true);

            // Lock changes may wake the body or clamp backend velocity immediately.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    for (handle, link, forces) in changed_forces.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            rb.reset_forces(true);
            rb.reset_torques(true);
            rb.add_force(forces.force, true);
            #[allow(clippy::useless_conversion)] // Need to convert if dim3 enabled
            rb.add_torque(forces.torque.into(), true);

            // Force changes can wake a body even when this frame performs no substep.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    for (handle, link, mut impulses) in changed_impulses.iter_mut() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            rb.apply_impulse(impulses.impulse, true);
            #[allow(clippy::useless_conversion)] // Need to convert if dim3 enabled
            rb.apply_torque_impulse(impulses.torque_impulse.into(), true);
            impulses.reset();

            // Impulses mutate backend velocity before the simulation boundary.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    for (handle, link, gravity_scale) in changed_gravity_scale.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            rb.set_gravity_scale(gravity_scale.0, true);

            // Gravity changes may wake the body before the next simulation boundary.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    for (handle, link, ccd) in changed_ccd.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            rb.enable_ccd(ccd.enabled);

            // Retain any activation transition caused by the CCD mode change.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    for (handle, link, soft_ccd) in changed_soft_ccd.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            rb.set_soft_ccd_prediction(soft_ccd.prediction);

            // Retain any activation transition caused by the prediction change.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    for (handle, link, dominance) in changed_dominance.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            rb.set_dominance_group(dominance.groups);

            // Retain any activation transition caused by the dominance change.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    for (handle, link, damping) in changed_damping.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle.0) {
            rb.set_linear_damping(damping.linear_damping);
            rb.set_angular_damping(damping.angular_damping);

            // Retain any activation transition caused by the damping change.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }

    for (handle, link, _) in changed_disabled.iter() {
        let rigidbody_set = rigid_body_sets
            .get_mut(link.0)
            .expect(RAPIER_CONTEXT_EXPECT_ERROR)
            .into_inner();
        if let Some(co) = rigidbody_set.bodies.get_mut(handle.0) {
            co.set_enabled(false);

            // Drain the disabled handle without querying or resurrecting its ECS entity.
            rigidbody_set.queue_body_for_writeback(handle.0);
        }
    }
}

/// System responsible for writing the result of the last simulation step into our `bevy_rapier`
/// components and the [`GlobalTransform`] component.
pub fn writeback_rigid_bodies(
    mut rigid_body_sets: Query<(Entity, &mut RapierRigidBodySet)>,
    timestep_mode: Res<TimestepMode>,
    config: Query<&RapierConfiguration>,
    sim_to_render_time: Query<&SimulationToRenderTime>,
    global_transforms: Query<&GlobalTransform>,
    mut writeback: Query<
        RigidBodyWritebackComponents,
        (With<RigidBody>, Without<RigidBodyDisabled>),
    >,
) {
    for (context_entity, mut rigid_body_set) in rigid_body_sets.iter_mut() {
        // Keep explicit changes queued while a paused context deliberately suppresses writeback.
        let config = config
            .get(context_entity)
            .expect("Could not get `RapierConfiguration`");
        if !config.physics_pipeline_active {
            rigid_body_set.writeback_stats = Default::default();
            continue;
        }

        // Resolve context-local interpolation state once before draining its reusable candidate list.
        let sim_to_render_time = sim_to_render_time
            .get(context_entity)
            .expect("Could not get `SimulationToRenderTime`");

        // Move the allocation out so field bookkeeping can mutate the context during the drain.
        let mut bodies_to_writeback = std::mem::take(&mut rigid_body_set.bodies_to_writeback);
        rigid_body_set.bodies_to_writeback_set.clear();
        let mut stats = crate::plugin::context::RigidBodyWritebackStats {
            visited: bodies_to_writeback.len(),
            ..Default::default()
        };

        for handle in bodies_to_writeback.iter().copied() {
            // Resolve the stable entity identity and snapshot backend output before borrowing ECS.
            let Some(rb) = rigid_body_set.bodies.get(handle) else {
                continue;
            };
            let entity = Entity::from_bits(rb.user_data as u64);
            let body_position = *rb.position();
            let body_velocity = Velocity {
                linear: rb.linvel(),
                #[cfg(feature = "dim3")]
                angular: rb.angvel(),
                #[cfg(feature = "dim2")]
                angular: rb.angvel(),
            };
            let body_sleeping = rb.is_sleeping();

            // Reject stale generations, context moves, disabled bodies, and removed ECS entities.
            let Ok((
                entity_handle,
                link,
                child_of,
                transform,
                mut interpolation,
                mut velocity,
                mut sleeping,
            )) = writeback.get_mut(entity)
            else {
                continue;
            };
            if entity_handle.0 != handle || link.0 != context_entity {
                continue;
            }
            stats.resolved += 1;

            // Reconstruct the render pose from the exact backend result and interpolation state.
            let mut entity_changed = false;
            let mut interpolated_pos = utils::iso_to_transform(&body_position);
            if let TimestepMode::Interpolated { dt, .. } = *timestep_mode {
                if let Some(interpolation) = interpolation.as_deref_mut() {
                    if interpolation.end.is_none() {
                        interpolation.end = Some(body_position);
                        entity_changed = true;
                    }

                    if let Some(interpolated) =
                        interpolation.lerp_slerp((dt + sim_to_render_time.diff) / dt)
                    {
                        interpolated_pos = utils::iso_to_transform(&interpolated);
                    }
                }
            }

            if let Some(mut transform) = transform {
                // Rapier stores collider scale rather than body scale, so preserve the ECS value.
                interpolated_pos = interpolated_pos.with_scale(transform.scale);

                // Parent-space reconstruction must use the live parent transform so the next
                // propagation produces the exact global value used for user-edit detection.
                if let Some(parent_global_transform) =
                    child_of.and_then(|c| global_transforms.get(c.parent()).ok())
                {
                    let (inverse_parent_scale, inverse_parent_rotation, inverse_parent_translation) =
                        parent_global_transform
                            .affine()
                            .inverse()
                            .to_scale_rotation_translation();
                    let new_rotation = inverse_parent_rotation * interpolated_pos.rotation;

                    #[allow(unused_mut)] // mut is needed in 2D but not in 3D.
                    let mut new_translation = inverse_parent_rotation
                        * inverse_parent_scale
                        * interpolated_pos.translation
                        + inverse_parent_translation;

                    // Preserve the user-owned depth coordinate in the 2D bridge.
                    #[cfg(feature = "dim2")]
                    {
                        new_translation.z = transform.translation.z;
                    }

                    // Avoid waking Bevy change detection when the published pose is unchanged.
                    if transform.rotation != new_rotation
                        || transform.translation != new_translation
                    {
                        transform.rotation = new_rotation;
                        transform.translation = new_translation;
                        entity_changed = true;
                    }

                    // Store the exact next propagated value so bridge output remains distinct from
                    // a subsequent user-authored GlobalTransform edit despite rounding.
                    let new_global_transform = parent_global_transform.mul_transform(*transform);
                    rigid_body_set
                        .last_body_transform_set
                        .insert(handle, new_global_transform);
                } else {
                    // Preserve the user-owned depth coordinate in the 2D bridge.
                    #[cfg(feature = "dim2")]
                    {
                        interpolated_pos.translation.z = transform.translation.z;
                    }

                    // Avoid waking Bevy change detection when the published pose is unchanged.
                    if transform.rotation != interpolated_pos.rotation
                        || transform.translation != interpolated_pos.translation
                    {
                        transform.rotation = interpolated_pos.rotation;
                        transform.translation = interpolated_pos.translation;
                        entity_changed = true;
                    }

                    // Preserve the private transform-edit discriminator on the active path.
                    rigid_body_set
                        .last_body_transform_set
                        .insert(handle, GlobalTransform::from(interpolated_pos));
                }
            }

            // Publish only backend velocity differences so settled candidates stay change-silent.
            if let Some(velocity) = &mut velocity {
                if **velocity != body_velocity {
                    **velocity = body_velocity;
                    entity_changed = true;
                }
            }

            // Publish the final active-to-sleep transition captured before the body left its island.
            if let Some(sleeping) = &mut sleeping {
                if sleeping.sleeping != body_sleeping {
                    sleeping.sleeping = body_sleeping;
                    entity_changed = true;
                }
            }

            if entity_changed {
                stats.changed += 1;
            }
        }

        // Reuse the drained allocation and expose cardinalities without a second world traversal.
        bodies_to_writeback.clear();
        rigid_body_set.bodies_to_writeback = bodies_to_writeback;
        rigid_body_set.writeback_stats = stats;
    }
}

/// System responsible for creating new Rapier rigid-bodies from the related `bevy_rapier` components.
pub fn init_rigid_bodies(
    mut commands: Commands,
    default_context_access: Query<Entity, With<DefaultRapierContext>>,
    mut rigidbody_sets: Query<(Entity, &mut RapierRigidBodySet)>,
    rigid_bodies: Query<RigidBodyComponents, Without<RapierRigidBodyHandle>>,
) {
    for (
        (entity, entity_context_link),
        rb,
        transform,
        vel,
        additional_mass_props,
        _mass_props,
        locked_axes,
        force,
        gravity_scale,
        (ccd, soft_ccd),
        dominance,
        sleep,
        damping,
        disabled,
        additional_solver_iters,
    ) in rigid_bodies.iter()
    {
        let mut builder = RigidBodyBuilder::new((*rb).into());
        builder = builder.enabled(disabled.is_none());

        if let Some(transform) = transform {
            builder = builder.pose(utils::transform_to_iso(&transform.compute_transform()));
        }

        #[allow(clippy::useless_conversion)] // Need to convert if dim3 enabled
        if let Some(vel) = vel {
            builder = builder.linvel(vel.linear.into()).angvel(vel.angular.into());
        }

        if let Some(locked_axes) = locked_axes {
            builder = builder.locked_axes((*locked_axes).into())
        }

        if let Some(gravity_scale) = gravity_scale {
            builder = builder.gravity_scale(gravity_scale.0);
        }

        if let Some(ccd) = ccd {
            builder = builder.ccd_enabled(ccd.enabled)
        }

        if let Some(soft_ccd) = soft_ccd {
            builder = builder.soft_ccd_prediction(soft_ccd.prediction)
        }

        if let Some(dominance) = dominance {
            builder = builder.dominance_group(dominance.groups)
        }

        if let Some(sleep) = sleep {
            builder = builder.sleeping(sleep.sleeping);
        }

        if let Some(damping) = damping {
            builder = builder
                .linear_damping(damping.linear_damping)
                .angular_damping(damping.angular_damping);
        }

        if let Some(mprops) = additional_mass_props {
            builder = match mprops {
                AdditionalMassProperties::MassProperties(mprops) => {
                    builder.additional_mass_properties(mprops.into_rapier())
                }
                AdditionalMassProperties::Mass(mass) => builder.additional_mass(*mass),
            };
        }

        if let Some(added_iters) = additional_solver_iters {
            builder = builder.additional_solver_iterations(added_iters.0);
        }

        builder = builder.user_data(entity.to_bits() as u128);

        let mut rb = builder.build();

        #[allow(clippy::useless_conversion)] // Need to convert if dim3 enabled
        if let Some(force) = force {
            rb.add_force(force.force.into(), false);
            rb.add_torque(force.torque.into(), false);
        }

        // NOTE: we can’t apply impulses yet at this point because
        //       the rigid-body’s mass isn’t up-to-date yet (its
        //       attached colliders, if any, haven’t been created yet).

        if let Some(sleep) = sleep {
            let activation = rb.activation_mut();
            activation.normalized_linear_threshold = sleep.normalized_linear_threshold;
            activation.angular_threshold = sleep.angular_threshold;
        }
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

        let Ok((_, mut rigidbody_set)) = rigidbody_sets.get_mut(context_entity) else {
            log::error!("Could not find entity {context_entity} with rapier context while initializing {entity}");
            continue;
        };
        let handle = rigidbody_set.bodies.insert(rb);
        // Bootstrap bodies can be fixed or sleeping and therefore never enter an active island.
        rigidbody_set.queue_body_for_writeback(handle);
        commands
            .entity(entity)
            .insert(RapierRigidBodyHandle(handle));
        rigidbody_set.entity2body.insert(entity, handle);

        if let Some(transform) = transform {
            rigidbody_set
                .last_body_transform_set
                .insert(handle, *transform);
        }
    }
}

/// This applies the initial impulse given to a rigid-body when it is created.
///
/// This cannot be done inside `init_rigid_bodies` because impulses require the rigid-body
/// mass to be available, which it was not because colliders were not created yet. As a
/// result, we run this system after the collider creation.
pub fn apply_initial_rigid_body_impulses(
    mut context: Query<(&mut RapierRigidBodySet, &RapierContextColliders)>,
    // We can’t use RapierRigidBodyHandle yet because its creation command hasn’t been
    // executed yet.
    mut init_impulses: Query<
        (Entity, &RapierContextEntityLink, &mut ExternalImpulse),
        Without<RapierRigidBodyHandle>,
    >,
) {
    for (entity, link, mut impulse) in init_impulses.iter_mut() {
        let (mut rigidbody_set, context_colliders) =
            context.get_mut(link.0).expect(RAPIER_CONTEXT_EXPECT_ERROR);
        let rigidbody_set = &mut *rigidbody_set;

        // Resolve the just-created handle before mutating its mass and velocity.
        let Some(handle) = rigidbody_set.entity2body.get(&entity).copied() else {
            continue;
        };
        if let Some(rb) = rigidbody_set.bodies.get_mut(handle) {
            // Make sure the mass-properties are computed.
            rb.recompute_mass_properties_from_colliders(&context_colliders.colliders);
            // Apply the impulse.
            rb.apply_impulse(impulse.impulse, false);

            #[allow(clippy::useless_conversion)] // Need to convert if dim3 enabled
            rb.apply_torque_impulse(impulse.torque_impulse.into(), false);

            impulse.reset();

            // Initial impulses mutate backend velocity even when this frame has no substep.
            rigidbody_set.queue_body_for_writeback(handle);
        }
    }
}

#[cfg(all(test, feature = "dim3"))]
mod tests {
    use super::*;
    use crate::plugin::context::{
        DefaultRapierContext, RapierContextJoints, RapierContextSimulation,
    };
    use crate::plugin::{NoUserData, RapierContextInitialization, RapierPhysicsPlugin};
    use bevy::time::{TimePlugin, TimeUpdateStrategy};
    use std::time::Duration;

    fn test_app(timestep_mode: TimestepMode, frame_duration: Duration) -> App {
        // Reproduce the production schedule with deterministic time and the smallest Bevy plugins.
        let mut app = App::new();
        app.add_plugins((
            TransformPlugin,
            TimePlugin,
            RapierPhysicsPlugin::<NoUserData>::default(),
        ));
        app.insert_resource(TimeUpdateStrategy::ManualDuration(frame_duration));
        app.insert_resource(timestep_mode);
        app.finish();
        app
    }

    fn two_context_test_app() -> (App, Entity, Entity) {
        // Disable automatic context creation so two context-local queues can be observed directly.
        let mut app = App::new();
        app.insert_resource(RapierContextInitialization::NoAutomaticRapierContext);
        app.add_plugins((
            TransformPlugin,
            TimePlugin,
            RapierPhysicsPlugin::<NoUserData>::default(),
        ));
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            1.0 / 60.0,
        )));
        app.finish();

        // Give each context an independent backend whose first body reuses the same raw handle.
        let context_a = app
            .world_mut()
            .spawn((
                RapierContextSimulation::default(),
                RapierConfiguration::new(1.0),
                DefaultRapierContext,
            ))
            .id();
        let context_b = app
            .world_mut()
            .spawn((
                RapierContextSimulation::default(),
                RapierConfiguration::new(1.0),
            ))
            .id();
        (app, context_a, context_b)
    }

    fn default_context_entity(app: &mut App) -> Entity {
        // Resolve the plugin-created context without assuming a stable Bevy entity generation.
        let world = app.world_mut();
        let mut query = world.query_filtered::<Entity, With<DefaultRapierContext>>();
        query.single(world).expect("default Rapier context")
    }

    fn assert_root_body_full_scan_parity(app: &mut App) {
        // Snapshot every enabled root body exactly as the legacy full-scan writeback would visit it.
        let rows = {
            let world = app.world_mut();
            let mut query = world.query_filtered::<(
                Entity,
                &RapierRigidBodyHandle,
                &RapierContextEntityLink,
                &Transform,
                &Velocity,
                &Sleeping,
            ), (
                With<RigidBody>,
                Without<RigidBodyDisabled>,
                Without<ChildOf>,
            )>();
            query
                .iter(world)
                .map(|(entity, handle, link, transform, velocity, sleeping)| {
                    (entity, handle.0, link.0, *transform, *velocity, *sleeping)
                })
                .collect::<Vec<_>>()
        };

        // Compare the optimized output for every legacy candidate with the exact backend state.
        for (entity, handle, context_entity, transform, velocity, sleeping) in rows {
            let rigidbody_set = app
                .world()
                .get::<RapierRigidBodySet>(context_entity)
                .expect("body context");
            let body = rigidbody_set
                .bodies
                .get(handle)
                .unwrap_or_else(|| panic!("backend body for {entity}"));
            let expected_transform = utils::iso_to_transform(body.position());
            approx::assert_relative_eq!(
                transform.translation,
                expected_transform.translation,
                epsilon = 1.0e-5
            );
            approx::assert_relative_eq!(
                transform.rotation,
                expected_transform.rotation,
                epsilon = 1.0e-5
            );
            approx::assert_relative_eq!(velocity.linear, body.linvel(), epsilon = 1.0e-5);
            approx::assert_relative_eq!(velocity.angular, body.angvel(), epsilon = 1.0e-5);
            assert_eq!(sleeping.sleeping, body.is_sleeping());
        }
    }

    #[test]
    fn direct_context_stepping_queues_active_results_once() {
        // Build the four context stores directly so this test cannot rely on the plugin's scheduled
        // wrapper to retain active bodies as a side effect.
        let mut simulation = RapierContextSimulation::default();
        let mut colliders = RapierContextColliders::default();
        let mut joints = RapierContextJoints::default();
        let mut rigidbody_set = RapierRigidBodySet::default();
        let handle = rigidbody_set.bodies.insert(
            RigidBodyBuilder::dynamic()
                .linvel(Vec3::X)
                .user_data(Entity::PLACEHOLDER.to_bits() as u128)
                .build(),
        );
        let mut time = Time::default();
        time.advance_by(Duration::from_secs_f32(1.0 / 60.0));
        let mut sim_to_render_time = SimulationToRenderTime::default();

        // Step through the public context method twice without draining writeback in between.
        for _ in 0..2 {
            simulation.step_simulation(
                &mut colliders,
                &mut joints,
                &mut rigidbody_set,
                Vec3::ZERO,
                TimestepMode::Variable {
                    max_dt: 1.0 / 60.0,
                    time_scale: 1.0,
                    substeps: 1,
                },
                None,
                &(),
                &time,
                &mut sim_to_render_time,
                None,
            );
        }

        // Direct stepping retains the result, and insertion-time dedup keeps one exact handle.
        assert_eq!(rigidbody_set.bodies_to_writeback.as_slice(), &[handle]);
        assert_eq!(rigidbody_set.bodies_to_writeback_set.len(), 1);
    }

    #[test]
    fn active_writeback_matches_full_scan_oracle_and_skips_settled_bodies() {
        let mut app = test_app(
            TimestepMode::Variable {
                max_dt: 1.0 / 60.0,
                time_scale: 1.0,
                substeps: 1,
            },
            Duration::from_secs_f32(1.0 / 60.0),
        );

        // Seed a settled resident set large enough to expose an accidental all-body traversal.
        let mut bodies = Vec::new();
        for index in 0..64 {
            let body = app
                .world_mut()
                .spawn((
                    RigidBody::Dynamic,
                    Collider::ball(0.25),
                    Transform::from_xyz(index as f32 * 2.0, 0.0, 0.0),
                    Velocity::default(),
                    Sleeping {
                        sleeping: true,
                        ..Default::default()
                    },
                ))
                .id();
            bodies.push(body);
        }

        // Drain bootstrap work, then require the unchanged frame to visit no resident body.
        app.update();
        app.update();
        let context_entity = default_context_entity(&mut app);
        assert_eq!(
            app.world()
                .get::<RapierRigidBodySet>(context_entity)
                .unwrap()
                .writeback_stats()
                .visited,
            0
        );
        assert_root_body_full_scan_parity(&mut app);

        // Wake one body through ordinary ECS inputs and compare all entities with the full-scan oracle.
        let active = bodies[17];
        app.world_mut()
            .entity_mut(active)
            .get_mut::<Sleeping>()
            .unwrap()
            .sleeping = false;
        app.world_mut()
            .entity_mut(active)
            .get_mut::<Velocity>()
            .unwrap()
            .linear = Vec3::X * 3.0;
        app.update();
        let stats = app
            .world()
            .get::<RapierRigidBodySet>(context_entity)
            .unwrap()
            .writeback_stats();
        assert_eq!(stats.resolved, stats.visited);
        assert!(stats.visited < bodies.len());
        assert_root_body_full_scan_parity(&mut app);

        // Remove motion and gravity first so prior-frame velocity feedback cannot wake the body
        // after the bridge deliberately processes sleeping changes before velocity changes.
        app.world_mut().entity_mut(active).insert(GravityScale(0.0));
        app.world_mut()
            .entity_mut(active)
            .get_mut::<Velocity>()
            .unwrap()
            .linear = Vec3::ZERO;
        app.update();

        // Publish the final active-to-sleep state on the next explicit transition frame.
        app.world_mut()
            .entity_mut(active)
            .get_mut::<Sleeping>()
            .unwrap()
            .sleeping = true;
        app.update();
        assert_root_body_full_scan_parity(&mut app);
        assert!(
            app.world()
                .entity(active)
                .get::<Sleeping>()
                .unwrap()
                .sleeping
        );
    }

    #[test]
    fn kinematic_inputs_write_back_during_a_zero_substep_frame() {
        let mut app = test_app(
            TimestepMode::Interpolated {
                dt: 1.0,
                time_scale: 1.0,
                substeps: 2,
            },
            Duration::from_secs_f32(0.25),
        );

        // Create one body for each kinematic mode and advance once to enter a render-only interval.
        let position_based = app
            .world_mut()
            .spawn((
                RigidBody::KinematicPositionBased,
                Transform::from_xyz(1.0, 0.0, 0.0),
                Velocity::default(),
                Sleeping::default(),
            ))
            .id();
        let velocity_based = app
            .world_mut()
            .spawn((
                RigidBody::KinematicVelocityBased,
                Transform::from_xyz(3.0, 0.0, 0.0),
                Velocity::default(),
                Sleeping::default(),
            ))
            .id();
        app.update();
        app.update();
        let context_entity = default_context_entity(&mut app);

        // Author both input forms while the interpolation accumulator guarantees no Rapier step.
        app.world_mut()
            .entity_mut(position_based)
            .get_mut::<Transform>()
            .unwrap()
            .translation = Vec3::new(2.0, 1.0, 0.0);
        app.world_mut()
            .entity_mut(velocity_based)
            .get_mut::<Velocity>()
            .unwrap()
            .linear = Vec3::new(-2.0, 0.5, 0.0);
        app.update();

        // The explicit-input candidates exactly match legacy output despite executing no substep.
        let stats = app
            .world()
            .get::<RapierRigidBodySet>(context_entity)
            .unwrap()
            .writeback_stats();
        assert_eq!(stats.visited, 2);
        assert_eq!(stats.resolved, 2);
        assert_root_body_full_scan_parity(&mut app);

        // A second render-only frame has neither new inputs nor interpolation opt-ins to publish.
        app.update();
        assert_eq!(
            app.world()
                .get::<RapierRigidBodySet>(context_entity)
                .unwrap()
                .writeback_stats()
                .visited,
            0
        );
    }

    #[test]
    fn explicit_body_input_families_deduplicate_to_one_candidate() {
        let mut app = test_app(
            TimestepMode::Interpolated {
                dt: 1.0,
                time_scale: 1.0,
                substeps: 2,
            },
            Duration::from_secs_f32(0.25),
        );

        // Seed every body-input family on one fixed body, then drain its bootstrap candidate.
        let body = app
            .world_mut()
            .spawn((
                RigidBody::Fixed,
                Transform::default(),
                Velocity::default(),
                AdditionalMassProperties::Mass(1.0),
                AdditionalSolverIterations(0),
                LockedAxes::empty(),
                ExternalForce::default(),
                ExternalImpulse::default(),
                GravityScale(1.0),
                Ccd::disabled(),
                SoftCcd::default(),
                Dominance::group(0),
                Damping::default(),
                Sleeping {
                    sleeping: true,
                    ..Default::default()
                },
            ))
            .id();
        app.update();
        app.update();
        let context_entity = default_context_entity(&mut app);

        // Change type, pose, velocity, mass, locks, force, impulse, gravity, CCD, dominance,
        // damping, solver iterations, and activation during one zero-substep frame.
        *app.world_mut()
            .entity_mut(body)
            .get_mut::<RigidBody>()
            .unwrap() = RigidBody::KinematicVelocityBased;
        app.world_mut()
            .entity_mut(body)
            .get_mut::<Transform>()
            .unwrap()
            .translation = Vec3::new(2.0, 3.0, 4.0);
        app.world_mut()
            .entity_mut(body)
            .get_mut::<Velocity>()
            .unwrap()
            .linear = Vec3::new(1.0, 2.0, 3.0);
        *app.world_mut()
            .entity_mut(body)
            .get_mut::<AdditionalMassProperties>()
            .unwrap() = AdditionalMassProperties::Mass(2.0);
        app.world_mut()
            .entity_mut(body)
            .get_mut::<AdditionalSolverIterations>()
            .unwrap()
            .0 = 3;
        *app.world_mut()
            .entity_mut(body)
            .get_mut::<LockedAxes>()
            .unwrap() = LockedAxes::TRANSLATION_LOCKED_X;
        app.world_mut()
            .entity_mut(body)
            .get_mut::<ExternalForce>()
            .unwrap()
            .force = Vec3::Y * 2.0;
        app.world_mut()
            .entity_mut(body)
            .get_mut::<ExternalImpulse>()
            .unwrap()
            .impulse = Vec3::X * 3.0;
        app.world_mut()
            .entity_mut(body)
            .get_mut::<GravityScale>()
            .unwrap()
            .0 = 0.5;
        app.world_mut()
            .entity_mut(body)
            .get_mut::<Ccd>()
            .unwrap()
            .enabled = true;
        app.world_mut()
            .entity_mut(body)
            .get_mut::<SoftCcd>()
            .unwrap()
            .prediction = 0.25;
        app.world_mut()
            .entity_mut(body)
            .get_mut::<Dominance>()
            .unwrap()
            .groups = 3;
        {
            let mut body_entity = app.world_mut().entity_mut(body);
            let mut damping = body_entity.get_mut::<Damping>().unwrap();
            damping.linear_damping = 0.1;
            damping.angular_damping = 0.2;
        }
        app.world_mut()
            .entity_mut(body)
            .get_mut::<Sleeping>()
            .unwrap()
            .sleeping = false;
        app.update();

        // Every explicit path converges on one handle and produces the legacy full-scan result.
        let stats = app
            .world()
            .get::<RapierRigidBodySet>(context_entity)
            .unwrap()
            .writeback_stats();
        assert_eq!(stats.visited, 1);
        assert_eq!(stats.resolved, 1);
        assert_root_body_full_scan_parity(&mut app);
    }

    #[test]
    fn parent_space_writeback_preserves_scale_and_edit_detection() {
        let mut app = test_app(
            TimestepMode::Variable {
                max_dt: 1.0 / 60.0,
                time_scale: 1.0,
                substeps: 1,
            },
            Duration::from_secs_f32(1.0 / 60.0),
        );

        // Build a scaled, rotated hierarchy around a fixed body so only explicit transforms can
        // select it for writeback.
        let parent = app
            .world_mut()
            .spawn(Transform::from_xyz(10.0, -2.0, 1.0).with_scale(Vec3::new(2.0, 3.0, 1.0)))
            .id();
        app.world_mut()
            .entity_mut(parent)
            .get_mut::<Transform>()
            .unwrap()
            .rotation = Quat::from_rotation_z(0.2);
        let child_scale = Vec3::new(4.0, 5.0, 6.0);
        let child = app
            .world_mut()
            .spawn((
                RigidBody::Fixed,
                Transform::from_xyz(1.0, 2.0, 0.5).with_scale(child_scale),
                Velocity::default(),
                Sleeping::default(),
                ChildOf(parent),
            ))
            .id();
        app.update();
        app.update();
        let context_entity = default_context_entity(&mut app);

        // Move the parent and child locally so the resulting GlobalTransform is genuinely authored.
        app.world_mut()
            .entity_mut(parent)
            .get_mut::<Transform>()
            .unwrap()
            .translation += Vec3::new(3.0, 1.0, -2.0);
        app.world_mut()
            .entity_mut(child)
            .get_mut::<Transform>()
            .unwrap()
            .translation = Vec3::new(-1.0, 1.5, 0.25);
        let authored_global =
            GlobalTransform::from(*app.world().entity(parent).get::<Transform>().unwrap())
                .mul_transform(*app.world().entity(child).get::<Transform>().unwrap());
        let expected_pose = utils::transform_to_iso(&authored_global.compute_transform());
        app.update();

        // The backend receives the authored global pose while parent-space reconstruction keeps
        // the ECS-local scale, translation, and writeback discriminator intact.
        let rigidbody_set = app
            .world()
            .get::<RapierRigidBodySet>(context_entity)
            .unwrap();
        let handle = rigidbody_set.entity2body()[&child];
        let backend_pose = rigidbody_set.bodies.get(handle).unwrap().position();
        approx::assert_relative_eq!(
            backend_pose.translation,
            expected_pose.translation,
            epsilon = 1.0e-5
        );
        approx::assert_relative_eq!(
            backend_pose.rotation,
            expected_pose.rotation,
            epsilon = 1.0e-5
        );
        assert_eq!(
            app.world().entity(child).get::<Transform>().unwrap().scale,
            child_scale
        );
        assert_eq!(rigidbody_set.writeback_stats().visited, 1);

        // The next unchanged frame must not misclassify bridge-authored hierarchy propagation as
        // another user edit, and a fixed body is absent from active-island fallback.
        app.update();
        assert_eq!(
            app.world()
                .get::<RapierRigidBodySet>(context_entity)
                .unwrap()
                .writeback_stats()
                .visited,
            0
        );
    }

    #[test]
    fn collider_and_joint_lifecycle_queue_exact_fixed_endpoints() {
        let mut app = test_app(
            TimestepMode::Variable {
                max_dt: 1.0 / 60.0,
                time_scale: 1.0,
                substeps: 1,
            },
            Duration::from_secs_f32(1.0 / 60.0),
        );

        // Fixed endpoints never enter Rapier's active island, isolating explicit collider and joint
        // lifecycle candidates from physics-driven candidates.
        let body_a = app
            .world_mut()
            .spawn((
                RigidBody::Fixed,
                Transform::default(),
                Velocity::default(),
                Sleeping::default(),
            ))
            .id();
        let body_b = app
            .world_mut()
            .spawn((
                RigidBody::Fixed,
                Transform::from_xyz(3.0, 0.0, 0.0),
                Velocity::default(),
                Sleeping::default(),
            ))
            .id();
        app.update();
        app.update();
        let context_entity = default_context_entity(&mut app);

        // Add, change, and remove an attached collider; each operation selects only its parent.
        let collider = app
            .world_mut()
            .spawn((Collider::ball(0.5), Transform::default(), ChildOf(body_a)))
            .id();
        app.update();
        assert_eq!(
            app.world()
                .get::<RapierRigidBodySet>(context_entity)
                .unwrap()
                .writeback_stats()
                .visited,
            1
        );
        *app.world_mut()
            .entity_mut(collider)
            .get_mut::<Collider>()
            .unwrap() = Collider::cuboid(0.5, 0.75, 1.0);
        app.update();
        assert_eq!(
            app.world()
                .get::<RapierRigidBodySet>(context_entity)
                .unwrap()
                .writeback_stats()
                .visited,
            1
        );
        app.world_mut().despawn(collider);
        app.update();
        assert_eq!(
            app.world()
                .get::<RapierRigidBodySet>(context_entity)
                .unwrap()
                .writeback_stats()
                .visited,
            1
        );

        // Add, change, and remove an impulse joint; every operation selects both exact endpoints.
        app.world_mut()
            .entity_mut(body_b)
            .insert(ImpulseJoint::new(body_a, FixedJointBuilder::new()));
        app.update();
        assert_eq!(
            app.world()
                .get::<RapierRigidBodySet>(context_entity)
                .unwrap()
                .writeback_stats()
                .visited,
            2
        );
        *app.world_mut()
            .entity_mut(body_b)
            .get_mut::<ImpulseJoint>()
            .unwrap() = ImpulseJoint::new(
            body_a,
            FixedJointBuilder::new().local_anchor1(Vec3::X * 0.25),
        );
        app.update();
        assert_eq!(
            app.world()
                .get::<RapierRigidBodySet>(context_entity)
                .unwrap()
                .writeback_stats()
                .visited,
            2
        );
        app.world_mut().entity_mut(body_b).remove::<ImpulseJoint>();
        app.update();
        assert_eq!(
            app.world()
                .get::<RapierRigidBodySet>(context_entity)
                .unwrap()
                .writeback_stats()
                .visited,
            2
        );
        assert_root_body_full_scan_parity(&mut app);
    }

    #[test]
    fn sleeping_interpolation_advances_across_render_only_frames() {
        let mut app = test_app(
            TimestepMode::Interpolated {
                dt: 1.0,
                time_scale: 1.0,
                substeps: 2,
            },
            Duration::from_secs_f32(0.25),
        );

        // Bootstrap a sleeping body at the final pose; Bevy's first manual-time update has zero
        // delta, and the next one lets Rapier classify it outside the active island.
        let body = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Collider::ball(0.25),
                Transform::from_xyz(4.0, 0.0, 0.0),
                TransformInterpolation::default(),
                Velocity::default(),
                GravityScale(0.0),
                Sleeping {
                    sleeping: true,
                    ..Default::default()
                },
            ))
            .id();
        app.update();
        app.update();
        let context_entity = default_context_entity(&mut app);
        let handle = app
            .world()
            .get::<RapierRigidBodySet>(context_entity)
            .unwrap()
            .entity2body()[&body];

        // Establish that active-island traversal alone cannot select the interpolation body.
        assert!(!app
            .world()
            .get::<RapierContextSimulation>(context_entity)
            .unwrap()
            .islands
            .active_bodies()
            .any(|active| active == handle));

        // Recreate a pending interpolation interval while leaving the sleeping backend at its
        // final pose; only the interpolation candidate path can now publish render progress.
        let backend_pose = *app
            .world()
            .get::<RapierRigidBodySet>(context_entity)
            .unwrap()
            .bodies
            .get(handle)
            .unwrap()
            .position();
        {
            let mut body_entity = app.world_mut().entity_mut(body);
            let mut interpolation = body_entity.get_mut::<TransformInterpolation>().unwrap();
            interpolation.start = Some(utils::transform_to_iso(&Transform::default()));
            interpolation.end = Some(backend_pose);
        }
        app.world_mut()
            .get_mut::<SimulationToRenderTime>(context_entity)
            .unwrap()
            .diff = -1.0;
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            0.25,
        )));

        // Consume four render-only frames and require monotonic progress to the sleeping pose.
        app.update();
        let first_x = app
            .world()
            .entity(body)
            .get::<Transform>()
            .unwrap()
            .translation
            .x;
        app.update();
        let second_x = app
            .world()
            .entity(body)
            .get::<Transform>()
            .unwrap()
            .translation
            .x;
        app.update();
        let third_x = app
            .world()
            .entity(body)
            .get::<Transform>()
            .unwrap()
            .translation
            .x;
        app.update();
        let fourth_x = app
            .world()
            .entity(body)
            .get::<Transform>()
            .unwrap()
            .translation
            .x;

        // Every frame visited the one opted-in sleeper without reactivating or moving its backend.
        let rigidbody_set = app
            .world()
            .get::<RapierRigidBodySet>(context_entity)
            .unwrap();
        assert!(second_x > first_x);
        assert!(third_x > second_x);
        assert!(fourth_x > third_x);
        approx::assert_relative_eq!(fourth_x, backend_pose.translation.x, epsilon = 1.0e-5);
        assert_eq!(rigidbody_set.writeback_stats().visited, 1);
        assert!(app.world().entity(body).get::<Sleeping>().unwrap().sleeping);
    }

    #[test]
    fn paused_context_reuses_storage_and_rejects_removed_handle_generations() {
        let (mut app, context_a, context_b) = two_context_test_app();

        // Bootstrap the first body in each context so their local handle values collide exactly.
        let body_a = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Transform::default(),
                Velocity::default(),
                Sleeping::default(),
                RapierContextEntityLink(context_a),
            ))
            .id();
        let body_b = app
            .world_mut()
            .spawn((
                RigidBody::Dynamic,
                Transform::default(),
                Velocity::default(),
                Sleeping::default(),
                RapierContextEntityLink(context_b),
            ))
            .id();
        app.update();
        let handle_a = app
            .world()
            .get::<RapierRigidBodySet>(context_a)
            .unwrap()
            .entity2body()[&body_a];
        let removed_handle = app
            .world()
            .get::<RapierRigidBodySet>(context_b)
            .unwrap()
            .entity2body()[&body_b];
        assert_eq!(handle_a, removed_handle);

        // Pause only B and repeat one explicit edit to prove pending work remains context-local and unique.
        app.world_mut()
            .get_mut::<RapierConfiguration>(context_b)
            .unwrap()
            .physics_pipeline_active = false;
        for speed in 1..=32 {
            app.world_mut()
                .entity_mut(body_b)
                .get_mut::<Velocity>()
                .unwrap()
                .linear = Vec3::X * speed as f32;
            app.update();
        }
        assert_eq!(
            app.world()
                .get::<RapierRigidBodySet>(context_b)
                .unwrap()
                .bodies_to_writeback
                .as_slice(),
            &[removed_handle]
        );

        // Remove the pending body while paused, then reuse its slot for a different generation.
        app.world_mut().despawn(body_b);
        app.update();
        assert!(app
            .world()
            .get::<RapierRigidBodySet>(context_b)
            .unwrap()
            .bodies_to_writeback
            .is_empty());
        let replacement = app
            .world_mut()
            .spawn((
                RigidBody::Fixed,
                Transform::default(),
                Velocity::linear(Vec3::X * 9.0),
                Sleeping {
                    sleeping: true,
                    ..Default::default()
                },
                RapierContextEntityLink(context_b),
            ))
            .id();
        app.update();
        let replacement_handle = app
            .world()
            .get::<RapierRigidBodySet>(context_b)
            .unwrap()
            .entity2body()[&replacement];
        let (removed_index, removed_generation) = removed_handle.into_raw_parts();
        let (replacement_index, replacement_generation) = replacement_handle.into_raw_parts();
        assert_eq!(removed_index, replacement_index);
        assert_ne!(removed_generation, replacement_generation);

        // Inject the stale generation beside real bootstrap work and resume only this context.
        {
            let mut rigidbody_set = app
                .world_mut()
                .get_mut::<RapierRigidBodySet>(context_b)
                .unwrap();
            rigidbody_set.queue_body_for_writeback(removed_handle);
        }
        app.world_mut()
            .get_mut::<RapierConfiguration>(context_b)
            .unwrap()
            .physics_pipeline_active = true;
        app.update();

        // The stale handle is visited but cannot resolve or overwrite the replacement entity.
        let rigidbody_set = app.world().get::<RapierRigidBodySet>(context_b).unwrap();
        let stats = rigidbody_set.writeback_stats();
        assert_eq!(stats.visited, 2);
        assert_eq!(stats.resolved, 1);
        assert!(rigidbody_set.bodies_to_writeback.is_empty());
        assert_eq!(
            app.world()
                .entity(replacement)
                .get::<Velocity>()
                .unwrap()
                .linear,
            Vec3::ZERO
        );
    }

    #[cfg(feature = "serde-serialize")]
    #[test]
    fn serialized_context_rebuilds_empty_transient_writeback_state() {
        // Populate durable Rapier state and every transient bridge index before serialization.
        let mut rigidbody_set = RapierRigidBodySet::default();
        let entity = Entity::PLACEHOLDER;
        let handle = rigidbody_set.bodies.insert(
            RigidBodyBuilder::fixed()
                .user_data(entity.to_bits() as u128)
                .build(),
        );
        rigidbody_set.entity2body.insert(entity, handle);
        rigidbody_set
            .last_body_transform_set
            .insert(handle, GlobalTransform::IDENTITY);
        rigidbody_set.queue_body_for_writeback(handle);
        rigidbody_set.writeback_stats = crate::plugin::context::RigidBodyWritebackStats {
            visited: 1,
            resolved: 1,
            changed: 1,
        };

        // Round-trip only the serialized contract; skipped indexes must default rather than retain
        // stale handles or diagnostics from the source world.
        let serialized = serde_json::to_string(&rigidbody_set).unwrap();
        let restored: RapierRigidBodySet = serde_json::from_str(&serialized).unwrap();
        assert_eq!(restored.bodies.len(), 1);
        assert!(restored.entity2body.is_empty());
        assert!(restored.last_body_transform_set.is_empty());
        assert!(restored.bodies_to_writeback.is_empty());
        assert!(restored.bodies_to_writeback_set.is_empty());
        assert_eq!(restored.writeback_stats(), Default::default());
    }
}
