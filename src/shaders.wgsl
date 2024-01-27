//		Initialization
//	Buffers to read/write
[[block]]
struct FluidState {
	position: array<vec2<f32>>;
	velocity: array<vec2<f32>>;

	predicted_pos: array<vec2<f32>>;

	density: array<vec2<f32>>;

	indices: array<uvec3>;
	offsets: array<u32>;
}

//	Settings
[[block]]
struct SimulationSettings {
	//		Particle tracking
	num_particles: u32;
	max_particles: u32;

	//		Timekeeping	
	dT: f32;

	//		Physics
	gravity: f32;

	prediction_factor: f32;

	smoothing_radius: f32;
	target_density: f32;
    
	pressure_multiplier: f32;
    pressure_multiplier_near: f32;

    viscosity_strength: f32;

	//		Bounds & obstacles
	bounds_size: vec2<f32>;
    obstacle_size: vec2<f32>;
    obstacle_pos: vec2<f32>;

	collision_damping: f32;
}

//	Bind groups
var<storage, read_write> fluidBuffers: FluidBuffers;

//Bind the buffers
[[binding(1), group(0)]]
var<storage, read_write> positions: array<vec2<f32>> @fluidBuffers.positions;

[[binding(2), group(0)]]
var<storage, read_write> velocities: array<vec2<f32>> @fluidBuffers.velocities;

[[binding(3), group(0)]]
var<storage, read_write> predictedPos: array<vec2<f32>> @fluidBuffers.predictedPos;

[[binding(4), group(0)]]
var<storage, read_write> densities: array<vec2<f32>> @fluidBuffers.densities;

[[binding(5), group(0)]]
var<storage, read_write> spatialIndices: array<uvec3> @fluidBuffers.spatialIndices;

[[binding(6), group(0)]]
var<storage, read_write> spatialOffsets: array<u32> @fluidBuffers.spatialOffsets;

//Bind the settings
[[binding(0), group(1)]]
var<storage, read> settings: SimulationSettings;


//		Define compute shaders
//	Position updating + collision handling
[[stage(compute), workgroup_size(64, 1, 1)]]
fn UpdatePositionsCompute(
	[[builtin(global_invocation_id)]] id: vec3<u32>
) {
    if (id.x >=  settings.num_particles) {
        return;
    }

    positions[id.x] += velocities[id.x] *  settings.dT;
    HandleCollisions(id.x);
}

fn HandleCollisions(
	i: u32,
) {
	//	Slice buffer data
    let mut pos = positions[i];
    let mut vel = velocities[i];

    //	Collide against bounds
    let bounds_half: vec2<f32> =  settings.bounds_size * 0.5;
    let bounds_dist: vec2<f32> = bounds_half - abs(pos);

    for (var axis: i32 = 0; axis < 2; axis = axis + 1) {
        if (bounds_dist[axis] <= 0.0) {
            pos[axis] = bounds_half[axis] * sign(pos[axis]);
            vel[axis] *= -1.0 *  settings.collision_damping;
        }
    }

    //	Collide against obstacle
    let obstacle_half =  settings.obstacle_size * 0.5;
    let obstacle_dist = obstacle_half - (pos -  settings.obstacle_pos).abs();

    if obstacle_dist.x >= 0. && obstacle_dist.y >= 0. {
        //	Determine the axis and offset using match
        let (axis, offset): (i32, f32) = match (obstacle_dist.x < obstacle_dist.y, obstacle_dist.x >= obstacle_dist.y) {
            (true, _) => (0,  settings.obstacle_pos.x),
            (_, true) => (1,  settings.obstacle_pos.y),
            _ => unreachable!(),
        };

        pos[axis] = obstacle_half[axis] * (pos[axis] - offset).signum() + offset;
        vel[axis] *= -1. *  settings.collision_damping;
    }

    //	Now update pos & vel
    positions[i] = pos;
    velocities[i] = vel;
}

//	External force handling
[[stage(compute), workgroup_size(64, 1, 1)]]
fn ApplyExternalForcesCompute(
	[[builtin(global_invocation_id)]] id: vec3<u32>
) {
    if (id.x >=  settings.num_particles) {
        return;
    }

	//	Slice data
	var pos: vec2<f32> = positions[id.x];
	var vel: vec2<f32> = velocities[id.x];

	//	Apply gravity
    let accel_gravity: vec2<f32> = vec2<f32>(0.0, settings.gravity);
	vel += accel_gravity * settings.dT;

	//	Update velocity
	velocities[id.x] = vel;
}

//	Prediction handling
[[stage(compute), workgroup_size(64, 1, 1)]]
fn UpdatePredictedPosCompute(
	[[builtin(global_invocation_id)]] id: vec3<u32>
) {
    if (id.x >=  settings.num_particles) {
        return;
    }

	//	Slice data
	var pos: vec2<f32> = positions[id.x];
	var vel: vec2<f32> = velocities[id.x];

    //	Predict
    predictedPos[id.x] = positions.position[id.x] + velocities.velocity[id.x] *  settings.prediction_factor;
}


//	Local coordinate hashing
[[stage(compute), workgroup_size(64, 1, 1)]]
fn UpdateSpatialHashCompute(
	[[builtin(global_invocation_id)]] id: vec3<u32>
) {
    if (id.x >= settings.num_particles) {
        return;
    }

    //	Reset offsets
    spatialOffsets[id.x] = settings.num_particles;

    //	Update index buffer
    let indices: vec2<i32> = Pos2Indices(predictedPos[id.x], settings.smoothing_radius);
    let hash: u32 = Indices2Hash(indices);
    let key: u32 = hash % settings.max_particles;
    
	spatialIndices[id.x] = uvec3(index, hash, key);
}

//Position to integer coordinates
fn Pos2Indices(position: vec2<f32>, radius: f32) -> vec2<i32> {
    return (position / radius).map<f32, i32>(floor);
}

//Integer coordinates to hash value
fn Indices2Hash(indices: vec2<i32>) -> u32 {
    let bitshift: u32 = 32 / indices.length();

    // Map indices to u32
    indices = indices.map<i32, u32>(cast);

    // Hash
    var hash: u32 = 0;
    for (var i: u32 = 0; i < indices.length(); i = i + 1) {
        var shifted: u32 = indices[i] << (i * bitshift);
        hash = hash ^ (shifted * HASH_KEYS[i]);
    }

    return hash;
}

//	Density computation
[[stage(compute), workgroup_size(64, 1, 1)]]
fn UpdateDensityCompute(
	[[builtin(global_invocation_id)]] id: vec3<u32>
) {
    if (id.x >= settings.num_particles) {
        return;
    }

    densities[id.x] = CalculateDensity(id.x);
}

fn CalculateDensity(
	i: u32,
) -> vec2<f32> {
	//	Retrieve position
	let pos: vec2<f32> = predictedPos[i];

	//	Precalculation
	let sqrRadius: f32 = settings.smoothing_radius * settings.smoothing_radius;

	//	Neighbor search - loop across local indices
	let originIndices: vec2<i32> = Pos2Indices(position, settings.smoothing_radius);

	var density: vec2<f32> = vec2<f32>(0.0, 0.0);
	for (var j: i32 = 0; j < 9; j = j + 1) {
		//	Find neighbor hash
		let hash: u32 = Indices2Hash(originIndices + OFFSETS_2D[j]);
		let key: u32 = hash % settings.max_particles;

		var k: u32 = localOffsets[key];
		while (k < settings.num_particles) {
			let indexData: uvec3 = localIndices[k];
			k = k + 1;

			//		Conditional checks
			//	Exit if no longer looking at the correct bin
			if (indexData.z != key) { break; }

			//	Skip if hash does not match
			if (indexData.y != hash) { continue; }

			//	Retrieve neighbor info
			let neighborIndex: u32 = indexData.x;

			//		Calculation
			//	Get vec2 from the predictedPositions buffer
			let neighborPos: vec2<f32> = predictedPos[neighborIndex];

			//	Calculate distance
			let neighborOffset: vec2<f32> = neighborPos - position;
			let neighborSqrDist: f32 = dot(neighborOffset, neighborOffset);

			//	Skip if not within radius (saves time)
			if (neighborSqrDist > sqrRadius) {
				continue;
			}

			//	Calculate density and near density
			let neighborDist: f32 = sqrt(neighborSqrDist);

			density.x += DensityKernel(neighborDist, settings.smoothing_radius);
			density.y += NearDensityKernel(neighborDist, settings.smoothing_radius);
		}
	}

	return density;
}

//	Pressure calculation
[[stage(compute), workgroup_size(64, 1, 1)]]
fn ApplyPressureForceCompute(
	[[builtin(global_invocation_id)]] id: vec3<u32>
) {
    if (id.x >= settings.num_particles) {
        return;
    }

    velocities[id.x] += CalculatePressure(id.x);
}
 
//	Note that the force component from pressure is directly proportional to the
//	density gradient, as per the Navier-Stokes equations. The relationship is
//	expressed by the equation:
//
//	F_pressure = -∇P
//
//	where:
//	F_pressure is the force from pressure,
//	∇P is the density gradient.
//
//	We calculate pressure at each particle by taking the difference between the
//	current density (from the buffer) and the desired resting density of the
//	fluid (from the settings). This is biased by a controllable multiplier. The 
//	pressure between the two particles is then averaged (to apply Newton's 3rd
//	Law). The final force component is then calculated as:
//	
//	direction * (pressure / density) * bias
//
//	where bias is inversely proportional to distance.
//
//	Some notes:
//	The mass of all particles is considered to be trivial, so no division by
//	mass is necessary for the completion of the equation. Instead, we divide
//	by the density at the point (since an instant of density is just mass).
//
//	There are two pressure forces calculated: one based on the difference from
//	the desired resting density of the fluid, and one based on only the density
//	at the particle. This second density (density_near) is necessary to act as
//	a repulsion force that keeps particles apart when everything is at the 
//	desired density - otherwise, particles glob together.

fn CalculatePressure(
	i: u32,
) -> vec2<f32> {
	//	Retrieve data from buffers
	let position: vec2<f32> = predictedPos[i];

	let density: f32 = densities[i][0];
	let densityNear: f32 = densities[i][1];

	//	Precalculation
	let sqrRadius: f32 = settings.smoothing_radius * settings.smoothing_radius;

	//	Calculate pressure at this particle
	let pressure: f32 = (density - settings.target_density) * settings.pressure_multiplier;
	let pressureNear: f32 = densityNear * settings.pressure_multiplier_near;

	//	Neighbor search - loop across local indices
	let originIndices: vec2<i32> = Pos2Indices(position, settings.smoothing_radius);

	var pressureForce: vec2<f32> = vec2<f32>(0.0, 0.0);
	for (var j: i32 = 0; j < 9; j = j + 1) {
		//	Find neighbor hash
		let hash: u32 = indicesToHash(originIndices + OFFSETS_2D[j]);
		let key: u32 = hash % settings.max_particles

		var k: u32 = localOffsets[key];
		while (k < settings.num_particles) {
			let indexData: uvec3 = localIndices[k];
			k = k + 1;

			//		Conditional checks
			//	Exit if no longer looking at the correct bin
			if (indexData.z != key) {
				break;
			}

			//	Skip if hash does not match
			if (indexData.y != hash) {
				continue;
			}

			//	Skip self
			let neighborIndex: u32 = indexData.x;
			if (neighborIndex == i) {
				continue;
			}

			//		Calculation
			//	Get vec2 from the predictedPositions buffer
			let neighborPos: vec2<f32> = predictedPositions[neighborIndex];

			//	Calculate distance
			let neighborOffset: vec2<f32> = neighborPos - position;
			let neighborSqrDist: f32 = dot(neighborOffset, neighborOffset);

			//	Skip if not within radius (saves time)
			if (neighborSqrDist > sqrRadius) {
				continue;
			}

			//	Calculate pressure
			let distance: f32 = sqrt(neighborSqrDist);
			let neighborDir: vec2<f32> = distance > 0.0 ? neighborOffset / distance : vec2<f32>(0.0, 0.0);

			let neighborDensity: f32 = densities[neighborIndex][0];
			let neighborDensityNear: f32 = densities[neighborIndex][1];

			let neighborPressure: f32 = (neighborDensity - settings.target_density) * settings.pressure_multiplier;
			let neighborPressureNear: f32 = neighborDensityNear * settings.pressure_multiplier_near;

			let sharedPressure: f32 = (pressure + neighborPressure) * 0.5;
			let sharedPressureNear: f32 = (pressureNear + neighborPressureNear) * 0.5;

			pressureForce += neighborDir
				* densityDerivative(distance, settings.smoothing_radius)
				* sharedPressure
				/ neighborDensity;

			pressureForce += neighborDir
				* nearDensityDerivative(distance, settings.smoothing_radius)
				* sharedPressureNear
				/ neighborDensityNear;
		}
	}

	//	Divide by density again (the mass part)
	return (pressureForce / density) * settings.dT;	
}

//	Viscosity calculation
[[stage(compute), workgroup_size(64, 1, 1)]]
fn ApplyViscosityForceCompute(
	[[builtin(global_invocation_id)]] id: vec3<u32>
) {
    if (id.x >= settings.num_particles) {
        return;
    }

    velocities[id.x] += CalculateViscosity(id.x);
}


//	Viscosity in a Newtonian fluid acts as, essentially, the diffusion of 
//	average velocity across the fluid. This is what the viscosity term in the
//	Navier-Stokes equations represents.
//
//	F_viscosity = mu * ∇^2 V
//
//	where:
//	F_viscosity is the force from viscosity,
//	∇V is the velocity gradient,
//	∇^2 V is the acceleration gradient.
//
//	We calculate acceleration from viscosity between each pair of particles by
//	looking at the difference in velocity between them, and adding some bias to
//	adjust for the distance between them. The end result ends up being
//	
//	(myVelocity - neighborVelocity) * bias * viscosityStrength
//
//	where bias is (again) inversely proportional to distance.
//
//	Some notes:
//	We only need to calculate the acceleration on the particle (because we 
//	assume that the mass of each particle is the same & is trivial), so no
//	division by mass (aka instantaneous density) is required.

fn CalculateViscosity(
	i: u32,
) -> vec2<f32> {
	//	Retrieve data from buffers
	let position: vec2<f32> = predictedPos[i];
	let velocity = velocities[i];

	let density: f32 = densities[i][0];
	let densityNear: f32 = densities[i][1];

	//	Precalculation
	let sqrRadius: f32 = settings.smoothing_radius * settings.smoothing_radius;

	// Neighbor search - loop across local indices
	let originIndices: vec2<i32> = Pos2Indices(position, settings.smoothing_radius);

	var viscosityForce: vec2<f32> = vec2<f32>(0.0, 0.0);
	for (var j: i32 = 0; j < 9; j = j + 1) {
		//	Find neighbor hash
		let hash: u32 = Indices2Hash(originIndices + OFFSETS_2D[j]);
		let key: u32 = hash % settings.max_particles;

		var k: u32 = spatialOffsets[key];
		while (k < settings.num_particles) {
			let indexData: uvec3 = spatialIndices[k];
			k = k + 1;

			//		Conditional checks
			//	Exit if no longer looking at the correct bin
			if (indexData.z != key) { break; }

			//	Skip if hash does not match
			if (indexData.y != hash) { continue; }

			//	Skip self
			let neighborIndex: u32 = indexData.x;
			if (neighborIndex == i) { continue; }

			//		Calculation
			//	Get vec2 from the predictedPositions buffer
			let neighborPos: vec2<f32> = predictedPositions[neighborIndex];

			//	Calculate distance
			let neighborOffset: vec2<f32> = neighborPos - position;
			let neighborSqrDist: f32 = dot(neighborOffset, neighborOffset);

			//	Skip if not within radius (saves time)
			if (neighborSqrDist > sqrRadius) {
				continue;
			}

			//	Calculate viscosity
			let distance: f32 = sqrt(neighborSqrDist);

			let neighborVelocity: vec2<f32> = velocities[neighborIndex];
			viscosityForce += (neighborVelocity - velocity) * viscosityKernel(distance, settings.smoothing_radius);
		}
	}

	//	Return adjusted viscosiy force
	return (viscosityForce * settings.viscosity_strength) * settings.dT;
}

//		Kernels
