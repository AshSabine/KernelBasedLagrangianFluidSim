//		Import
use wgpu::{
	util::DeviceExt,
	Device
};
use bytemuck;

use nalgebra::{Vector2, Point2};

mod hashing;
use hashing::*;



//		Structures
#[derive(Copy, Clone, Debug)]
struct Fluidistanceate {	//	SoA because this allows me to have seperate pos and vel buffers
    pos: Vec<Vector2<f32>>,
    vel: Vec<Vector2<f32>>,
}

#[derive(Copy, Clone)]
struct SimulationSettings {
	//		Timekeeping	
	time_scale: f32,
	framerate: i32,

	//		Physics
	gravity: f32,

	prediction_factor: f32,

	smoothing_radius: f32,
	target_density: f32,
    
	pressure_multiplier: f32,
    pressure_multiplier_near: f32,

    viscosity_strength: f32,

	//		Bounds & obstacles
	bounds_size: Vector2<f32>,
    obstacle_size: Vector2<f32>,
    obstacle_pos: Vector2<f32>,

	collision_damping: f32,
}

struct FluidSimulation {
	num_particles: u32,
	particles: Vec<Particle>,

	//	Buffers
	position_buffer: wgpu::Buffer<f32>,		//	Vector2<f32>
	velocity_buffer: wgpu::Buffer<f32>,		//	Vector2<f32>

	predicted_pos_buffer: wgpu::Buffer<f32>,	//	Vector2<f32>

	density_buffer:	wgpu::Buffer<f32>,		//	f32

	local_indices: wgpu::Buffer<f32>,			//	Vector3<u32>
	local_offsets: wgpu::Buffer<f32>,			//	u32

	//	Params
	settings: SimulationSettings,

	//	Executor
	device: Context,
}

impl FluidSimulation {
	fn new(device: Device, window: &mut PistonWindow) -> Self {
		//		Create buffers
		//	Basic data (pos, vel)
        let position_buffer = create_buffer_helper(device, &self.particles.pos);
        let velocity_buffer = create_buffer_helper(device, &self.particles.vel);

		//	
        Simulation {
            particles: Vec::new(),
            
			position_buffer,
            velocity_buffer,
            // ... Initialize other buffers

            context,
            settings: SimulationSettings {
                // Initialize your simulation settings
                // ...
            },
            is_paused: false,
            spawn_data: spawner.get_spawn_data(),
            pause_next_frame: false,
            num_particles: spawn_data.positions.len() as i32,
        }
    }

	fn run_simulation_step(&mut self) {
        //	Update particle positions. This is the first thing we do.

		//	Apply external forces to the particles.
	
		//	Predict where the particles are going to be next frame.

		//	Update the spacial hash.

		//	Update all densities.

		//	Calculate pressure force

		//	Calculate viscosity force
	}
}

//		Kernel level bs for GPU accel
//	Kernel-building macro.
macro_rules! auto_build_kernel {
	($context:expr, $device:expr, $function:ident, $($arg:expr),*) => {
        $context
			.config_for_device($device).unwrap()		//	Configure for specified device & unwrap
            .create_kernel_builder::<$function>()		//	Build the kernel for the specified func
            .expect("Failed to create kernel builder")	//	Handle errors
            $(.arg(&$arg))*								//	Add args onto the kernel
            .launch((self.num_particles, 1, 1))			//	Launch
            .is_ok()									//	Return success/failure
	}
}

//	Updates all the positions & then does collision handling.
unsafe fn update_positions_kernel(
	positions: &mut Vector2<f32>, //[f32], 
	velocities: &Vector2<f32>, //[f32],
	settings: &SimulationSettings,
) {
	/*
	let i = thread::index_1d() as usize;

	//	Slice the data from the buffer
	let position_slice = &mut positions[i..i + 2];
	let velocity_slice = &velocities[i..i + 2];

	//	Update data
	position_slice.iter_mut()
		.zip(velocity_slice.iter())
		.for_each(|(pos, vel)| {
        	*pos += *vel;
    });
	*/

	for (pos, vel) in positions.iter_mut().zip(velocities.iter()) {
        pos += vel
    }

	//	Handle collisions
	handle_collisions_kernel(positions, velocities, settings);
}

//	Collision handling w/ border & obstacle
unsafe fn handle_collisions_kernel(
	positions: &mut [f32],
    velocities: &mut [f32],
	settings: &SimulationSettings,
) {
	let i = thread::index_1d() as usize * 2;

	//	Slice the data from the buffer
	let mut pos = Vector2::new(positions.get(i), positions.get(i + 1));
    let mut vel = Vector2::new(velocities.get(i), velocities.get(i + 1));

	//	Collide against bounds
	let bounds_half = settings.bounds_size * 0.5;
	let bounds_dist = bounds_half - pos.abs();

	for axis in 0..2 {
		if bounds_dist[axis] <= 0. {
			pos[axis] = bounds_half[axis] * pos[axis].signum();
			vel[axis] *= -1. * settings.collision_damping;
		}
	}

	//	Collide against obstacle
	let obstacle_half = settings.obstacle_size * 0.5;
	let obstacle_dist = obstacle_half - (pos - settings.obstacle_pos).abs();

	if obstacle_dist.x >= 0. && obstacle_dist.y >= 0. {
		//	I wanted to use a for loop here to make it dimension-extendable, 
		//	but I don't know how to have it check every other. I think I'm
		//	going to end up using a match.
        let (axis, offset) = if obstacle_dist.x < obstacle_dist.y {
			(0, settings.obstacle_pos.x)
		} else {
			(1, settings.obstacle_pos.y)
		};
	
		pos[axis] = obstacle_half[axis] * (pos[axis] - offset).signum() + offset;
		vel[axis] *= -1. * settings.collision_damping;
    }

	//	Now update pos & vel
	positions.set(i, pos.x);
	positions.set(i + 1, pos.y);
	velocities.set(i, vel.x);
	velocities.set(i + 1, vel.y);
}

//	Apply any external forces
unsafe fn external_forces_kernel(
	positions: &mut [f32],
    velocities: &mut [f32],
	settings: &SimulationSettings,
) {
	let i = thread::index_1d() as usize * 2;

	//	Slice the data from the buffer
	let position_slice = &mut positions[i..i + 2];
	let velocity_slice = &mut velocities[i..i + 2];

	//	Apply gravity
	let accel_gravity = Vector2::new(0., settings.gravity);
	velocity_slice.iter_mut()
		.zip(accel_gravity.iter())
		.for_each(|(vel, accel)| {
        	*vel += *accel;
    });
}

//	Calculating the predicted positions next frame.
unsafe fn predict_position_kernel(
	positions: &[f32],
    velocities: &[f32],
	predicted_positions: &mut [f32],
	settings: &SimulationSettings,
) {
	let i = thread::index_1d() as usize * 2;

	//	Slice the data from the buffer
	let position_slice = &positions[i..i + 2];
	let velocity_slice = &velocities[i..i + 2];

	let predicted_slice = &mut predicted_positions[i..i + 2];

	//	Apply prediction
	predicted_slice.iter_mut()
		.zip(position_slice.iter().zip(velocity_slice.iter()))
		.for_each(|(predicted, (pos, vel))| {
        	*predicted = *pos + *vel * settings.prediction_factor;
    });
}


unsafe fn update_density_kernel(
    predicted_positions: &[f32],
	densities: &[f32],
    local_indices: &[u32],
    local_offsets: &[u32],
	settings: &SimulationSettings,
	num_particles: u32,
) {
	let i = thread::index_1d() as usize;

	//	Precalculation
	let sqr_radius = settings.smoothing_radius * settings.smoothing_radius;

	//	Slice data
	let position_slice = &predicted_positions[i*2..i*2 + 2];
	let position = Vector2::from_slice_unaligned_unchecked(position_slice);

	//	Neighbor search - loop across local indices
	let origin_indices = pos_to_indices(position, settings.smoothing_radius);

    let mut density = 0.;
	let mut density_near = 0.;
	for j in 0..9 {
		//	Find neighbor hash
		let hash = indices_to_hash(origin_indices + OFFSETS_2D[j]);
		let key = hash_to_key(hash, num_particles);

		let k = local_offsets[key];
		while k < num_particles {
			let index_data = &local_indices[k*3..k*3 + 3];
			k += 1;
	
			//		Conditional checks
			//	Exit if no longer looking at the correct bin
			if index_data[2] != key { break; }
	
			//	Skip if hash does not match
			if index_data[1] != hash { continue; }
	
			//		Calculation
			//	Retrieve neighbor info	
			let neighbor_index = index_data[0];
			
			let neighbor_slice = &positions[neighbor_index*2..neighbor_index*2 + 2];
			let neighbor_pos = Vector2::from_slice_unaligned_unchecked(neighbor_slice);

			//	Calculate distance
			let neighbor_offset = neighbor_predicted_pos - position;
			let neighbor_sqr_dist = dot(neighbor_offset, neighbor_offset);
	
			//	Skip if not within radius (saves time)
			if neighbor_sqr_dist > sqr_radius {
				continue;
			}
	
			//	Calculate density and near density
			let neighbor_dist = neighbor_sqr_dist.sqrt();
			
			density += DensityKernel(distance, settings.smoothing_radius);
			density_near += NearDensityKernel(distance, settings.smoothing_radius);
		}
	}

	//	Set
	densities[i*2] = density;
	densities[i*2 + 1] = density_near;
}

unsafe fn calculate_pressure_kernel(
	predicted_positions: &[f32],
    densities: &[f32],
    velocities: &mut [f32],
	local_indices: &[u32],
    local_offsets: &[u32],
	settings: &SimulationSettings,
	num_particles: u32,
) {
	let i = thread::index_1d() as usize * 2;

	//	Slice data
	let density = densities[i];
	let density_near = densities[i+1];

	let position_slice = &predicted_positions[i..i + 2];
	let position = Vector2::from_slice_unaligned_unchecked(position_slice);
	
	//	Precalculation
	let sqr_radius = settings.smoothing_radius * settings.smoothing_radius;

	let pressure = (density - settings.target_density) * settings.pressure_multiplier;
	let pressure_near = density_near * settings.pressure_multiplier_near;

	//	Neighbor search - loop across local indices
	let origin_indices = pos_to_indices(position, settings.smoothing_radius);

	let mut pressure_force = 0.;
	for j in 0..9 {
		//	Find neighbor hash
		let hash = indices_to_hash(origin_indices + OFFSETS_2D[j]);
		let key = hash_to_key(hash, num_particles);

		let k = local_offsets[key];
		while k < num_particles {
			let index_data = &local_indices[k*3..k*3 + 3];
			k += 1;
	
			//		Conditional checks
			//	Exit if no longer looking at the correct bin
			if index_data[2] != key { break; }
	
			//	Skip if hash does not match
			if index_data[1] != hash { continue; }
	
			//		Calculation
			//	Retrieve neighbor info	
			let neighbor_index = index_data[0];

			//	Skip self
			if neighbour_index == i { continue; }
			
			let neighbor_slice = &positions[neighbor_index*2..neighbor_index*2 + 2];
			let neighbor_pos = Vector2::from_slice_unaligned_unchecked(neighbor_slice);

			//	Calculate distance
			let neighbor_offset = neighbor_predicted_pos - position;
			let neighbor_sqr_dist = dot(neighbor_offset, neighbor_offset);
	
			//	Skip if not within radius (saves time)
			if neighbor_sqr_dist > sqr_radius { continue; }
	
			//	Calculate pressure
			let distance = neighbor_sqr_dist.sqrt();
            let neighbor_dir = if distance > 0.0 {
                offset_to_neighbour / distance
            } else {  Vector2::zeros() };

            let neighbour_density = densities[neighbour_index][0];
            let neighbour_density_near = densities[neighbour_index][1];
			
            let neighbour_pressure = (neighbour_density - settings.target_density) * settings.pressure_multiplier;;
            let neighbor_pressure_near = neighbour_density_near * settings.pressure_multiplier_near;

            let shared_pressure = (pressure + neighbour_pressure) * 0.5;
            let shared_pressure_near = (near_pressure + neighbor_pressure_near) * 0.5;

            pressure_force += dir_to_neighbour
                * density_derivative(distance, smoothing_radius)
                * shared_pressure
                / neighbour_density;

            pressure_force += dir_to_neighbour
                * near_density_derivative(distance, smoothing_radius)
                * shared_pressure_near
                / neighbour_density_near;
		}
	}

	let acceleration = pressure_force / density;
    velocities[i] += acceleration.x;
    velocities[i + 1] += acceleration.y;
}

//	Calculate viscosity force
unsafe fn calculate_viscosity_kernel(
    predicted_positions: &[f32],
    velocities: &mut [f32],
    spatial_indices: &[u32],
    spatial_offsets: &[u32],
    settings: &SimulationSettings,
    num_particles: u32,
) {
	let i = thread::index_1d() as usize * 2;

	//	Slice data
	let density = densities[i];
	let density_near = densities[i+1];

	let position_slice = &predicted_positions[i..i + 2];
	let position = Vector2::from_slice_unaligned_unchecked(position_slice);

	let velocity_slice = &velocities[i..i + 2];
	let velocity = Vector2::from_slice_unaligned_unchecked(velocity_slice);

	//	Precalculation
	let sqr_radius = settings.smoothing_radius * settings.smoothing_radius;

	//	Neighbor search - loop across local indices
	let origin_indices = pos_to_indices(position, settings.smoothing_radius);

	let mut viscosity_force = 0.;
	for j in 0..9 {
		//	Find neighbor hash
		let hash = indices_to_hash(origin_indices + OFFSETS_2D[j]);
		let key = hash_to_key(hash, num_particles);

		let k = local_offsets[key];
		while k < num_particles {
			let index_data = &local_indices[k*3..k*3 + 3];
			k += 1;
	
			//		Conditional checks
			//	Exit if no longer looking at the correct bin
			if index_data[2] != key { break; }
	
			//	Skip if hash does not match
			if index_data[1] != hash { continue; }
	
			//		Calculation
			//	Retrieve neighbor info	
			let neighbor_index = index_data[0];

			//	Skip self
			if neighbour_index == i { continue; }
			
			let neighbor_slice = &positions[neighbor_index*2..neighbor_index*2 + 2];
			let neighbor_pos = Vector2::from_slice_unaligned_unchecked(neighbor_slice);

			//	Calculate distance
			let neighbor_offset = neighbor_predicted_pos - position;
			let neighbor_sqr_dist = dot(neighbor_offset, neighbor_offset);
	
			//	Skip if not within radius (saves time)
			if neighbor_sqr_dist > sqr_radius { continue; }
	
			//	Calculate viscocity
			let distance = neighbor_sqr_dist.sqrt();

            let neighbour_velocity_slice = &velocities[i*2..i*2 + 2];
			let neighbour_velocity = Vector2::from_slice_unaligned_unchecked(neighbour_velocity_slice);

            viscosity_force += (neighbour_velocity - velocity) * ViscosityKernel(distance, settings.smoothing_radius);
		}
	}

	let acceleration = viscosity_force * settings.viscosity_strength;
    velocities[i] += acceleration.x;
    velocities[i + 1] += acceleration.y;
}