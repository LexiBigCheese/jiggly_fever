#![no_std]

use core::ops::{Add, Mul};

extern crate alloc;

pub struct PhysicsProperties {
    pub gravity: f32,
    pub velocity_to_impact: f32,
    pub min_impactable: f32,
    pub jiggle_stiff: f32,
    pub jiggle_damp: f32,
    pub jiggle_life_decrease_rate: f32,
    pub jiggle_life_threshold: f32,
    pub jiggle_life_threshold_inverse: f32,
    pub jiggle_offset_epsilon: f32,
    pub jiggle_momentum_epsilon: f32,
}

pub enum SlimeState {
    Settled,
    Falling {
        velocity: f32,
    },
    Jiggling {
        momentum: f32,
        offset: f32,
        life: f32,
    },
}

pub struct SlimePropsIn {
    pub state: SlimeState,
    pub y_bottom: f32,
}

pub struct SlimePropsOut {
    pub state: SlimeState,
    pub y_bottom: f32,
    pub y_scale: f32,
    pub x_scale: f32,
}

pub trait MutSlime {
    fn modify_slime(&mut self, f: impl FnOnce(SlimePropsIn) -> SlimePropsOut);
}

pub trait JiggleImpulsable {
    fn modify_state(&mut self, f: impl FnOnce(SlimeState) -> SlimeState);
}

pub trait Direction {
    fn other_directions(self) -> impl Iterator<Item = Self>;
    fn opposite(self) -> Self;
    const UP: Self;
}

pub struct JigglePropagation<Loc, Dir> {
    pub at: Loc,
    pub impulse: f32,
    pub came_from: Dir,
}

pub trait JigglyBoard {
    type Dir: Direction + Copy + Clone;
    type Loc: Copy + Clone;
    ///This acts as the "Jiggle Allowed in Direction", "Get new Location", and "Jiggle Falloff/Transfer" function
    fn apply_dir_to_loc(
        &self,
        dir: Self::Dir,
        loc: Self::Loc,
        impulse: f32,
    ) -> Option<(Self::Loc, f32)>;
    fn cols(&self) -> impl Iterator<Item = impl Iterator<Item = Self::Loc>>;
    fn mut_slime_with(&mut self, loc: Self::Loc, f: impl FnOnce(SlimePropsIn) -> SlimePropsOut);
    fn impulse_jiggle_with(&mut self, loc: Self::Loc, f: impl FnOnce(SlimeState) -> SlimeState);

    /// If this returns true, the board is settled
    fn run_physics(&mut self, dt: f32, physprop: &PhysicsProperties) -> bool {
        let mut jiggle_propagations: alloc::vec::Vec<JigglePropagation<Self::Loc, Self::Dir>> =
            alloc::vec![];
        let mut settled = true;
        let cols = self
            .cols()
            .map(|col| col.enumerate().collect::<alloc::vec::Vec<_>>())
            .collect::<alloc::vec::Vec<_>>();
        for col in cols {
            let mut jiggle_offset = 0.0;
            for (y, location) in col {
                self.mut_slime_with(location, |SlimePropsIn { state, y_bottom }| {
                    use SlimeState::*;
                    match state {
                        Settled => {
                            let out = SlimePropsOut {
                                state,
                                y_bottom: jiggle_offset,
                                y_scale: 1.0,
                                x_scale: 1.0,
                            };
                            jiggle_offset += 1.0;
                            out
                        }
                        Falling { velocity } => {
                            settled = false;
                            let velocity = velocity + dt * physprop.gravity;
                            let y_bottom = y_bottom - velocity * dt;
                            if y_bottom <= jiggle_offset {
                                jiggle_propagations.push(JigglePropagation {
                                    at: location,
                                    impulse: physprop.velocity_to_impact * velocity,
                                    came_from: Self::Dir::UP,
                                });
                                let y_bottom = jiggle_offset;
                                jiggle_offset += 1.0;
                                SlimePropsOut {
                                    state: Jiggling {
                                        momentum: 0.0,
                                        offset: 0.0,
                                        life: 1.0,
                                    },
                                    x_scale: 1.0,
                                    y_scale: 1.0,
                                    y_bottom,
                                }
                            } else {
                                let clamped_vel = velocity.mul(1.0 / 9.0).add(1.0).clamp(1.0, 2.0);
                                let x_scale = 1.0 / clamped_vel;
                                let y_scale = 1.0 * clamped_vel;
                                SlimePropsOut {
                                    state: Falling { velocity },
                                    y_bottom,
                                    y_scale,
                                    x_scale,
                                }
                            }
                        }
                        Jiggling {
                            momentum,
                            offset,
                            life,
                        } => {
                            settled = false;
                            let y_bottom = jiggle_offset;
                            let accdt = physprop.jiggle_stiff * -offset * dt;
                            let mut momentum = (momentum + accdt) * physprop.jiggle_damp;
                            let mut offset = offset + momentum * dt;
                            if life < physprop.jiggle_life_threshold {
                                offset *= life * physprop.jiggle_life_threshold_inverse;
                                momentum *= life * physprop.jiggle_life_threshold_inverse;
                            }
                            if (life <= 0.0)
                                || (offset.abs() < physprop.jiggle_offset_epsilon
                                    && momentum.abs() < physprop.jiggle_momentum_epsilon)
                            {
                                jiggle_offset += 1.0;
                                SlimePropsOut {
                                    state: Settled,
                                    y_bottom,
                                    y_scale: 1.0,
                                    x_scale: 1.0,
                                }
                            } else {
                                let life = life - physprop.jiggle_life_decrease_rate * dt;
                                let y_scale = (1.0 - offset).max(0.0);
                                let x_scale = y_scale.max(0.5).recip();
                                jiggle_offset += y_scale;
                                SlimePropsOut {
                                    state: Jiggling {
                                        momentum,
                                        offset,
                                        life,
                                    },
                                    x_scale,
                                    y_scale,
                                    y_bottom,
                                }
                            }
                        }
                    }
                });
            }
        }
        //Now run through jiggle propagations
        for propagation in jiggle_propagations {
            self.propagate_jiggle(propagation, physprop);
        }
        settled
    }
    fn propagate_jiggle(
        &mut self,
        propagation: JigglePropagation<Self::Loc, Self::Dir>,
        physprop: &PhysicsProperties,
    ) {
        let JigglePropagation {
            at,
            impulse,
            came_from,
        } = propagation;
        let PhysicsProperties {
            min_impactable,
            velocity_to_impact,
            ..
        } = physprop;
        if impulse < *min_impactable {
            return;
        }
        use SlimeState::*;

        self.impulse_jiggle_with(at, |state| {
            match state {
                Settled => Jiggling {
                    momentum: impulse,
                    offset: 0.0,
                    life: 1.0,
                },
                //Note: this really should not be encountered, but it will have defined behaviour in the case it is.
                Falling { velocity } => Jiggling {
                    momentum: impulse + velocity * *velocity_to_impact,
                    offset: 0.0,
                    life: 1.0,
                },
                Jiggling {
                    momentum, offset, ..
                } => Jiggling {
                    momentum: momentum + impulse,
                    offset,
                    life: 1.0,
                },
            }
        });

        for dir in came_from.other_directions() {
            let Some((at, impulse)) = self.apply_dir_to_loc(dir, at, impulse) else {
                continue;
            };
            self.propagate_jiggle(
                JigglePropagation {
                    at,
                    impulse,
                    came_from: dir.opposite(),
                },
                physprop,
            );
        }
    }
}
