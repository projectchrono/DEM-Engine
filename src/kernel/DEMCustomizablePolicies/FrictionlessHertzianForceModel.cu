// DEM force calculation strategies, modifiable

// Material properties and time (user referrable)
float E_cnt, CoR_cnt;
{
    mat2ProxyContactParam<float>(E_cnt, CoR_cnt, E[bodyAMatType], nu[bodyAMatType], CoR[bodyAMatType], E[bodyBMatType],
                                 nu[bodyBMatType], CoR[bodyBMatType]);
}

float3 rotVelCPA, rotVelCPB;
{
    // We also need the relative velocity between A and B in global frame to use in the damping terms
    // To get that, we need contact points' rotational velocity in GLOBAL frame
    // This is local rotational velocity (the portion of linear vel contributed by rotation)
    rotVelCPA = cross(ARotVel, locCPA);
    rotVelCPB = cross(BRotVel, locCPB);
    // This is mapping from local rotational velocity to global
    applyOriQToVector3<float, smug::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AoriQw, AoriQx, AoriQy, AoriQz);
    applyOriQToVector3<float, smug::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BoriQw, BoriQx, BoriQy, BoriQz);
}

// The (total) relative linear velocity of A relative to B
const float3 velB2A = (ALinVel + rotVelCPA) - (BLinVel + rotVelCPB);
const float projection = dot(velB2A, B2A);

const float mass_eff = (AOwnerMass * BOwnerMass) / (AOwnerMass + BOwnerMass);
float sqrt_Rd = sqrt(overlapDepth * (ARadius * BRadius) / (ARadius + BRadius));
const float Sn = 2. * E_cnt * sqrt_Rd;

const float loge = (CoR_cnt < SMUG_DEM_TINY_FLOAT) ? log(SMUG_DEM_TINY_FLOAT) : log(CoR_cnt);
float beta = loge / sqrt(loge * loge + smug::PI_SQUARED);

const float k_n = smug::TWO_OVER_THREE * Sn;
const float gamma_n = smug::TWO_TIMES_SQRT_FIVE_OVER_SIX * beta * sqrt(Sn * mass_eff);

// normal force (that A feels)
force += (k_n * overlapDepth + gamma_n * projection) * B2A;