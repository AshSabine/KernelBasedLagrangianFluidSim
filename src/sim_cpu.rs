//		Import
use nalgebra::{Vector2, Point2};

mod hashing;
use hashing::*;

//		Data
const OFFSETS_2D: [Vector2<i32>; 9] = [
	Vector2::new(-1, -1),
	Vector2::new(0, -1),
	Vector2::new(1, -1),
	Vector2::new(-1, 0),
	Vector2::new(0, 0),
	Vector2::new(1, 0),
	Vector2::new(-1, 1),
	Vector2::new(0, 1),
	Vector2::new(1, 1),
];

const HASH_KEYS: [u32; 3] = [
	2938, 9537247, 4281573253
];

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
	position_buffer: Vec<Vector2<f32>>,		//	Vector2<f32>
	velocity_buffer: Vec<Vector2<f32>>,		//	Vector2<f32>

	predicted_pos_buffer: Vec<Vector2<f32>>,	//	Vector2<f32>

	density_buffer:	Vec<Vector2<f32>>,		//	f32

	local_indices: Vec<Vector3<u32>>,			//	Vector3<u32>
	local_offsets: Vec<u32>,			//	u32

	//	Params
	settings: SimulationSettings,
}

impl FluidSimulation {
	fn new(device: Device, window: &mut PistonWindow) -> Self {
		//		Create buffers
		//	Basic data (pos, vel)
		let position_buffer = initial_state.pos.clone();
		let velocity_buffer = initial_state.vel.clone();

		//	Other physics
		let predicted_pos_buffer = vec![Vector2::zeros(); MAX_PARTICLES];
		let density_buffer = vec![0.0; MAX_PARTICLES];

		//	Spacial hashing stuff
		let local_indices_buffer = vec![Vector3::zeros(); MAX_PARTICLES];
		let local_offsets_buffer = vec![0; MAX_PARTICLES];

		//	Settings
		let settings = Settings {
			max_particles: MAX_PARTICLES,

			// Timekeeping
			time_scale: 1.0,
			framerate: 30,

			// Physics
			gravity: 9.81,

			prediction_factor: 1.0 / 120.0,

			smoothing_radius: 2.0,
			target_density: 60.0,

			pressure_multiplier: 0.5,
			pressure_multiplier_near: 0.5,

			viscosity_strength: 0.1,

			// Bounds & obstacles
			bounds_size: Vector2::new(800.0, 600.0),
			obstacle_size: Vector2::new(100.0, 100.0),
			obstacle_pos: Vector2::new(400.0, 200.0),

			collision_damping: 0.5,
		};

		Self {
			num_particles: 1000 as u32,

			// Simulation data
			// Buffers
			position_buffer,
			velocity_buffer,

			predicted_pos_buffer,
			density_buffer,

			local_indices_buffer,
			local_offsets_buffer,

			// Other
			settings,

			// Rendering data
			vertex_buffer,
		}
	}

	pub fn compute(&mut self) {
		update_positions();
		apply_external_forces();
		predict_next_positions();
		update_locality();
		update_densities();
		apply_pressure();
		apply_viscosity();
	}
}

//		For loops
//	Updates all the positions & then does collision handling.
fn update_positions(
	positions: &mut Vec<Vector2<f32>>,
	velocities: &Vec<Vector2<f32>>,
	settings: &SimulationSettings,
) {
	for (pos, vel) in positions.iter_mut().zip(velocities.iter()) {
		*pos += *vel;
	}

	// Handle collisions
	handle_collisions(positions, velocities, settings);
}

//	Collision handling w/ border & obstacle
fn handle_collisions(
	positions: &mut Vec<f32>,
	velocities: &mut Vec<f32>,
	settings: &SimulationSettings,
) {
	for (pos, vel) in positions.iter_mut().zip(velocities.iter_mut()) {
		// Collide against bounds
		let bounds_half = settings.bounds_size * 0.5;
		let bounds_dist = bounds_half - pos.abs();

		for axis in 0..2 {
			if bounds_dist[axis] <= 0. {
				pos[axis] = bounds_half[axis] * pos[axis][axis].signum();
				vel[axis] *= -1. * settings.collision_damping;
			}
		}

		// Collide against obstacle
		let obstacle_half = settings.obstacle_size * 0.5;
		let obstacle_dist = obstacle_half - (pos - settings.obstacle_pos).abs();

		if obstacle_dist.x >= 0. && obstacle_dist.y >= 0. {
			// Use a for loop to find the minimum distance and corresponding axis
			let (axis, offset) = (0..2)
				.map(|axis| (axis, settings.obstacle_pos[axis]))
				.min_by(|a, b| {
					obstacle_dist[a.0]
						.partial_cmp(&obstacle_dist[b.0])
						.unwrap_or(std::cmp::Ordering::Equal)
				})
				.unwrap();

			pos[axis] = obstacle_half[axis] * (pos[axis][axis] - offset).signum() + offset;
			vel[axis] *= -1. * settings.collision_damping;
		}
	}
}

//	Apply any external forces
fn external_forces(
	positions: &mut Vec<Vector2<f32>>,
	velocities: &mut Vec<Vector2<f32>>,
	settings: &SimulationSettings,
) {
	for (pos, vel) in positions.iter_mut().zip(velocities.iter_mut()) {
		// Apply gravity
		vel.y += settings.gravity;
	}
}

//	Calculating the predicted positions next frame.
fn predict_position(
	positions: &Vec<Vector2<f32>>,
	velocities: &Vec<Vector2<f32>>,
	predicted_positions: &mut Vec<Vector2<f32>>,
	settings: &SimulationSettings,
) {
	for ((pos, vel), predicted) in positions
		.iter()
		.zip(velocities.iter())
		.zip(predicted_positions.iter_mut())
	{
		// Apply prediction
		*predicted = *pos + *vel * settings.prediction_factor;
	}
}


fn update_density(
	predicted_positions: &Vec<Vector2<f32>>,
	densities: &mut Vec<Vector2<f32>>,
	local_indices: &Vec<Vector3<u32>>,
	local_offsets: &Vec<u32>,
	settings: &SimulationSettings,
	num_particles: u32,
) {
	// Precalculation
	let sqr_radius = settings.smoothing_radius * settings.smoothing_radius;

	for i in 0..num_particles as usize {
		// Slice data
		let position = predicted_positions[i];

		// Neighbor search - loop across local indices
		let origin_indices = pos_to_indices(position, settings.smoothing_radius);

		let mut density = 0.;
		let mut density_near = 0.;
		for j in 0..9 {
			// Find neighbor hash
			let hash = indices_to_hash(origin_indices + OFFSETS_2D[j]);
			let key = hash % MAX_PARTICLES;

			let mut k = local_offsets[key] as usize;
			while k < num_particles as usize {
				let index_data = &local_indices[k];
				k += 1;

				// Conditional checks
				// Exit if no longer looking at the correct bin
				if index_data.z != key {
					break;
				}

				// Skip if hash does not match
				if index_data.y != hash {
					continue;
				}

				// Calculation
				// Retrieve neighbor info
				let neighbor_index = index_data.x as usize;
				let neighbor_pos = predicted_positions[neighbor_index];

				// Calculate distance
				let neighbor_offset = neighbor_pos - position;
				let neighbor_sqr_dist = dot(neighbor_offset, neighbor_offset);

				// Skip if not within radius (saves time)
				if neighbor_sqr_dist > sqr_radius {
					continue;
				}

				// Calculate density and near density
				let neighbor_dist = neighbor_sqr_dist.sqrt();

				density += density_kernel(neighbor_dist, settings.smoothing_radius);
				density_near += near_density_kernel(neighbor_dist, settings.smoothing_radius);
			}
		}

		// Set
		densities[i].x = density;
		densities[i].y = density_near;
	}
}

fn calculate_pressure(
	predicted_positions: &Vec<Vector2<f32>>,
	densities: &Vec<Vector2<f32>>,
	velocities: &mut Vec<f32>,
	local_indices: &Vec<Vector3<u32>>,
	local_offsets: &Vec<u32>,
	settings: &SimulationSettings,
	num_particles: u32,
) {
	// Precalculation
	let sqr_radius = settings.smoothing_radius * settings.smoothing_radius;

	for i in 0..num_particles as usize {
		// Slice data
		let density = densities[i].x;
		let density_near = densities[i].y;

		let position = predicted_positions[i];

		let pressure = (density - settings.target_density) * settings.pressure_multiplier;
		let pressure_near = density_near * settings.pressure_multiplier_near;

		// Neighbor search - loop across local indices
		let origin_indices = pos_to_indices(position, settings.smoothing_radius);

		let mut pressure_force = Vector2::zeros();
		for j in 0..9 {
			// Find neighbor hash
			let hash = indices_to_hash(origin_indices + OFFSETS_2D[j]);
			let key = hash % MAX_PARTICLES;

			let mut k = local_offsets[key] as usize;
			while k < num_particles as usize {
				let index_data = &local_indices[k];
				k += 1;

				// Conditional checks
				// Exit if no longer looking at the correct bin
				if index_data.z != key {
					break;
				}

				// Skip if hash does not match
				if index_data.y != hash {
					continue;
				}

				// Calculation
				// Retrieve neighbor info
				let neighbor_index = index_data.x as usize;

				// Skip self
				if neighbor_index == i {
					continue;
				}

				let neighbor_pos = predicted_positions[neighbor_index];

				// Calculate distance
				let neighbor_offset = neighbor_pos - position;
				let neighbor_sqr_dist = dot(neighbor_offset, neighbor_offset);

				// Skip if not within radius (saves time)
				if neighbor_sqr_dist > sqr_radius {
					continue;
				}

				// Calculate pressure
				let distance = neighbor_sqr_dist.sqrt();
				let neighbor_dir = if distance > 0.0 {
					neighbor_offset / distance
				} else {
					Vector2::zeros()
				};

				let neighbour_density = densities[neighbor_index][0];
				let neighbour_density_near = densities[neighbor_index][1];

				let neighbour_pressure =
					(neighbour_density - settings.target_density) * settings.pressure_multiplier;
				let neighbor_pressure_near = neighbour_density_near * settings.pressure_multiplier_near;

				let shared_pressure = (pressure + neighbour_pressure) * 0.5;
				let shared_pressure_near = (pressure_near + neighbor_pressure_near) * 0.5;

				pressure_force +=
					neighbor_dir * density_derivative(distance, settings.smoothing_radius)
						* shared_pressure
						/ neighbour_density;

				pressure_force +=
					neighbor_dir * density_derivative_near(distance, settings.smoothing_radius)
						* shared_pressure_near
						/ neighbour_density_near;
			}
		}

		let acceleration = pressure_force / density;
		velocities[i].x += acceleration.x;
		velocities[i].y += acceleration.y;
	}
}

//	Calculate viscosity force
fn calculate_viscosity(
	predicted_positions: &Vec<Vector2<f32>>,
	velocities: &mut Vec<Vector2<f32>>,
	spatial_indices: &Vec<Vector3<u32>>,
	spatial_offsets: &Vec<u32>,
	settings: &SimulationSettings,
	num_particles: u32,
) {
	// Precalculation
	let sqr_radius = settings.smoothing_radius * settings.smoothing_radius;

	for i in 0..num_particles as usize {
		// Slice data
		let density = densities[i].x;
		let density_near = densities[i].x;

		let position = predicted_positions[i];

		let velocity = &mut velocities[i];

		// Neighbor search - loop across local indices
		let origin_indices = pos_to_indices(position, settings.smoothing_radius);

		let mut viscosity_force = Vector2::zeros();
		for j in 0..9 {
			// Find neighbor hash
			let hash = indices_to_hash(origin_indices + OFFSETS_2D[j]);
			let key = hash % MAX_PARTICLES;

			let mut k = spatial_offsets[key] as usize;
			while k < num_particles as usize {
				let index_data = &spatial_indices[k];
				k += 1;

				// Conditional checks
				// Exit if no longer looking at the correct bin
				if index_data.z != key {
					break;
				}

				// Skip if hash does not match
				if index_data.y != hash {
					continue;
				}

				// Calculation
				// Retrieve neighbor info
				let neighbor_index = index_data.x as usize;

				// Skip self
				if neighbor_index == i {
					continue;
				}

				let neighbor_pos = predicted_positions[neighbor_index];

				// Calculate distance
				let neighbor_offset = neighbor_pos - position;
				let neighbor_sqr_dist = dot(neighbor_offset, neighbor_offset);

				// Skip if not within radius (saves time)
				if neighbor_sqr_dist > sqr_radius {
					continue;
				}

				// Calculate viscosity
				let distance = neighbor_sqr_dist.sqrt();

				let neighbour_velocity = &velocities[neighbor_index];

				viscosity_force +=
					(neighbour_velocity - velocity) * viscosity_kernel(distance, settings.smoothing_radius);
			}
		}

		let acceleration = viscosity_force * settings.viscosity_strength;
		velocity.x += acceleration.x;
		velocity.y += acceleration.y;
	}
}

//	Helper funcs
fn pos_to_indices(position: Vector2<f32>, radius: f32) -> Vector2<i32> {
	Vector2::new((position.x / radius).floor() as i32, (position.y / radius).floor() as i32)
}

fn indices_to_hash(indices: Vector2<i32>) -> u32 {
	const BIT_SHIFT: u32 = 32 / 2;
	let mut hash: u32 = 0;

	// Map indices to u32
	let indices_unsigned = Vector2::new(indices.x as u32, indices.y as u32);

	// Hash
	for i in 0..3 {
		let shifted = indices_unsigned[i] << (i * BIT_SHIFT);
		hash ^= shifted * HASH_KEYS[i];
	}

	hash
}

//		Kernels
//	Internal
fn density_kernel(d: f32, r: f32) -> f32 { spike2(d, r) }
fn near_density_kernel(d: f32, r: f32) -> f32 { spike3(d, r) }
fn density_derivative(d: f32, r: f32) -> f32 { derivative2(d, r) }
fn density_derivative_near(d: f32, r: f32) -> f32 { derivative3(d, r) }
fn viscosity_kernel(d: f32, r: f32) -> f32 { spike2(d, r) }

//	Math
fn spike3(d: f32, r: f32) -> f32 {
	if d < r {
		let v = r - d;
		return v * v * v * 0.5;
	}
	0.
}

fn spike2(d: f32, r: f32) -> f32 {
	if d < r {
		let v = r - d;
		return v * v * v * 0.5;
	}
	0.
}

fn derivative3(d: f32, r: f32) -> f32 {
	if d <= r {
		let v = r - d;
		return -v * v * 0.5;
	}
	0.
}

fn derivative2(d: f32, r: f32) -> f32 {
	if d <= r {
		let v = r - d;
		return -v * 0.5;
	}
	0.
}