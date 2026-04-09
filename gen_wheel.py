import argparse, sys, os, gc
import rot_geo_gen as wg

# Parse input arguments
parser = argparse.ArgumentParser(description=__doc__, formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--cp_deviation", "-c", type=str, dest="nls", default="variational",
#                     choices=["variational", "rmturs"],help="Navier--Stokes non-linear solver, rmturs or FEniCS variational solver")
parser.add_argument("--density","-d", type=int, dest="g_density", default=12,
                    help="Number of grousers for one revolution of the wheel")
parser.add_argument("--wave_num","-n", type=int, dest="g_period", default=3,
                    help="The wave number for one grouser (number of periods)")
parser.add_argument("--rad", "-r", type=float, dest="rad", default=0.25,
                    help="Wheel's radius (excluding grousers)")
parser.add_argument("--width", "-w", type=float, dest="width", default=0.2,
                    help="Wheel's width")
parser.add_argument("--cp_deviation", "-c", type=float, dest="cp_deviation", default=0.,
                    help="Wheel's outer perimeter control points' vertical deviation distance")
parser.add_argument("--height", "-g", type=float, dest="g_height", default=0.02,
                    help="Height of the grousers")
parser.add_argument("--thickness", "-t", type=float, dest="g_width", default=0.015,
                    help="Width (thickness) of the grousers")
parser.add_argument("--amp", "-a", type=float, dest="g_amp", default=0.03,
                    help="Amplitude of the grousers (as they have sinusoidal shapes)")
parser.add_argument("--curved", "-s", type=bool, dest="g_curved", default=False,
                    help="Grouser pattern shape: True if sinusoidal, False if straight")
parser.add_argument("--out_file", "-f", type=str, dest="outfile", default="wheel.obj",
                    help="Output obj file name")
parser.add_argument("--tri_count", "-u", type=int, dest="tri_count", default=50000,
                    help="Target number of triangles")

args = parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    mesh = wg.GenWheel(rad=args.rad, width=args.width, cp_deviation=args.cp_deviation, 
                       g_height=args.g_height, g_width=args.g_width, g_density=args.g_density, 
                       g_amp=args.g_amp, g_period=args.g_period, g_curved=args.g_curved,
                       filename=args.outfile,tri_count=args.tri_count)
