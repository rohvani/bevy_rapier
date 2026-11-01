use super::{GlobalTransform, RigidBodyHandle};

/// Previous bridge output, indexed by the context's backend arena slot.
/// Only live backend handles may be inserted. Removed generations cannot read or erase a new body,
/// and the separate occupancy bit leaves every u32 generation valid, including u32::MAX.
#[derive(Default, Clone)]
pub(crate) struct BodyTransformCache {
    slots: Vec<BodyTransformSlot>,
}

#[derive(Default, Clone)]
struct BodyTransformSlot {
    transform: GlobalTransform,
    generation: u32,
    occupied: bool,
}

impl BodyTransformCache {
    pub(crate) fn get(&self, handle: &RigidBodyHandle) -> Option<&GlobalTransform> {
        let (index, generation) = handle.into_raw_parts();
        let slot = self.slots.get(index as usize)?;
        (slot.occupied && slot.generation == generation).then_some(&slot.transform)
    }

    pub(crate) fn insert(&mut self, handle: RigidBodyHandle, transform: GlobalTransform) {
        // Live Rapier indices bound retained storage by the backend arena's high-water mark.
        let (index, generation) = handle.into_raw_parts();
        let index = index as usize;
        if self.slots.len() <= index {
            self.slots
                .resize_with(index + 1, BodyTransformSlot::default);
        }
        self.slots[index] = BodyTransformSlot {
            transform,
            generation,
            occupied: true,
        };
    }

    pub(crate) fn remove(&mut self, handle: &RigidBodyHandle) -> Option<GlobalTransform> {
        let (index, generation) = handle.into_raw_parts();
        let slot = self.slots.get_mut(index as usize)?;
        if !slot.occupied || slot.generation != generation {
            return None;
        }
        slot.occupied = false;
        Some(slot.transform)
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.slots.iter().all(|slot| !slot.occupied)
    }
}

#[cfg(test)]
mod body_transform_cache_tests {
    use super::super::Transform;
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn live_generations_match_a_handle_map_through_reuse_and_stale_removal() {
        let mut cache = BodyTransformCache::default();
        let mut oracle = HashMap::new();
        let mut retired = Vec::new();
        let mut active = Vec::new();
        // Exercise noncontiguous slots and the entire generation domain, without a sentinel.
        for generation in [0, 1, u32::MAX, 7] {
            for old in active.drain(..) {
                assert_eq!(cache.remove(&old), oracle.remove(&old));
                retired.push(old);
            }
            for index in [0, 17, 3, 5] {
                let handle = RigidBodyHandle::from_raw_parts(index, generation);
                let pose = GlobalTransform::from(Transform::from_xyz(
                    index as f32,
                    generation as f32,
                    2.0,
                ));
                oracle.insert(handle, pose);
                cache.insert(handle, pose);
                active.push(handle);
            }
            for handle in active.iter().chain(&retired) {
                assert_eq!(cache.get(handle), oracle.get(handle));
            }
            for old in &retired {
                assert_eq!(cache.remove(old), None);
            }
            for handle in &active {
                assert_eq!(cache.get(handle), oracle.get(handle));
            }
        }
        let invalid = RigidBodyHandle::from_raw_parts(u32::MAX, u32::MAX);
        assert_eq!(cache.get(&invalid), None);
        assert_eq!(cache.remove(&invalid), None);
        assert_eq!(cache.slots.len(), 18);
        let cloned = cache.clone();
        for handle in active {
            assert_eq!(cache.remove(&handle), oracle.remove(&handle));
            assert!(cloned.get(&handle).is_some());
            assert_eq!(cache.get(&handle), None);
        }
        assert!(cache.is_empty());
    }
}
