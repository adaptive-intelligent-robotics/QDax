from qdax.tasks.qd_suite.archimedean_spiral import (
    ArchimedeanDescriptor,
    ArchimedeanSpiralV0,
    ParameterizationGenotype,
)
from qdax.tasks.qd_suite.deceptive_evolvability import DeceptiveEvolvabilityV0
from qdax.tasks.qd_suite.ssf import SsfV0

archimedean_spiral_v0_angle_euclidean_task = ArchimedeanSpiralV0(
    ParameterizationGenotype.angle,
    ArchimedeanDescriptor.euclidean,
)
archimedean_spiral_v0_angle_geodesic_task = ArchimedeanSpiralV0(
    ParameterizationGenotype.angle,
    ArchimedeanDescriptor.geodesic,
)
archimedean_spiral_v0_arc_length_euclidean_task = ArchimedeanSpiralV0(
    ParameterizationGenotype.arc_length,
    ArchimedeanDescriptor.euclidean,
)
archimedean_spiral_v0_arc_length_geodesic_task = ArchimedeanSpiralV0(
    ParameterizationGenotype.arc_length,
    ArchimedeanDescriptor.geodesic,
)
deceptive_evolvability_v0_task = DeceptiveEvolvabilityV0()
ssf_v0_param_size_1_task = SsfV0(param_size=1)
ssf_v0_param_size_2_task = SsfV0(param_size=2)
