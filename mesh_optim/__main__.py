"""
CLI entry point for mesh optimization package
"""

import argparse
import sys
import logging
import json
from pathlib import Path

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mesh_optim.log')
        ]
    )

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AortaCFD Mesh Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1: Geometry-driven mesh optimization
  python -m mesh_optim stage1 --geometry cases_input/patient1 --config mesh_optim/configs/stage1_default.json

  # Stage 2: GCI-based WSS convergence verification (literature-backed)
  python -m mesh_optim stage2 --geometry cases_input/patient1 --model RANS

  # Stage 2: LES verification (requires high-end hardware)
  python -m mesh_optim stage2 --geometry cases_input/patient1 --model LES

  # Stage 2: Laminar verification (for low Reynolds number cases)
  python -m mesh_optim stage2 --geometry cases_input/patient1 --model LAMINAR
        """
    )
    
    subparsers = parser.add_subparsers(dest='stage', help='Optimization stage')
    
    # Stage 1 subparser
    stage1_parser = subparsers.add_parser('stage1', help='Geometry-driven mesh optimization')
    stage1_parser.add_argument('--geometry', required=True, help='Path to geometry directory (with STL files)')
    stage1_parser.add_argument('--config', help='Configuration file (default: stage1_default.json)')
    stage1_parser.add_argument('--output', help='Output directory (default: output/<patient>/meshOptimizer/stage1)')
    stage1_parser.add_argument('--max-iterations', type=int, default=4, help='Maximum iterations')
    stage1_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Stage 2 subparser
    stage2_parser = subparsers.add_parser('stage2', help='GCI-based WSS convergence verification')
    stage2_parser.add_argument('--geometry', required=True, help='Path to geometry directory')
    stage2_parser.add_argument('--model', choices=['LAMINAR', 'RANS', 'LES'], default='RANS', help='Flow model')
    stage2_parser.add_argument('--config', help='Configuration file (default: auto-select based on model)')
    stage2_parser.add_argument('--output', help='Output directory (default: output/<patient>/meshOptimizer/stage2_<model>)') 
    stage2_parser.add_argument('--max-iterations', type=int, default=3, help='Maximum QoI iterations')
    stage2_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    stage2_parser.add_argument('--skip-cfd', action='store_true', help='Skip CFD simulations (mesh-only mode)')
    stage2_parser.add_argument('--conservative', action='store_true', help='Use conservative settings (prevents crashes)')
    stage2_parser.add_argument('--max-memory', type=float, help='Maximum memory limit in GB (auto-detected if not set)')
    
    args = parser.parse_args()
    
    if not args.stage:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger('mesh_optim.cli')
    
    try:
        if args.stage == 'stage1':
            from .stage1_mesh import Stage1MeshOptimizer
            
            # Use default config if not specified
            if not args.config:
                args.config = Path(__file__).parent / "configs" / "stage1_default.json"
            
            # Generate default output path if not specified
            if not args.output:
                patient_name = Path(args.geometry).name
                args.output = Path("output") / patient_name / "meshOptimizer" / "stage1"
            
            logger.info(f"Starting Stage 1 optimization: {args.geometry}")
            logger.info(f"Output directory: {args.output}")
            optimizer = Stage1MeshOptimizer(args.geometry, args.config, args.output)
            
            if args.max_iterations:
                optimizer.max_iterations = args.max_iterations
                
            result_dir = optimizer.iterate_until_quality()
            logger.info(f"✅ Stage 1 completed: {result_dir}")
            
        elif args.stage == 'stage2':
            from .stage2_gci import Stage2GCIVerifier
            
            # Check system resources and warn user
            import psutil
            memory_gb = psutil.virtual_memory().available / (1024**3)
            cpu_count = psutil.cpu_count()
            
            logger.info(f"🖥️  System: {memory_gb:.1f}GB RAM, {cpu_count} CPUs")
            if memory_gb < 4:
                logger.warning("⚠️ Low memory detected - consider using --conservative flag")
            
            # Generate default output path if not specified
            if not args.output:
                patient_name = Path(args.geometry).name
                model_suffix = args.model.lower()
                args.output = Path("output") / patient_name / "meshOptimizer" / f"stage2_{model_suffix}"
            
            logger.info(f"Starting Stage 2 GCI verification: {args.geometry} ({args.model})")
            
            # Find Stage 1 best mesh
            patient_name = Path(args.geometry).name
            stage1_best = Path("output") / patient_name / "meshOptimizer" / "stage1" / "best"
            
            if not stage1_best.exists():
                logger.error(f"❌ Stage 1 mesh not found: {stage1_best}")
                logger.info("💡 Run Stage 1 first: python -m mesh_optim stage1 --geometry tutorial/patient1")
                return 1
            
            # Load configuration
            if not args.config:
                args.config = Path(__file__).parent / "configs" / "stage1_default.json"
            
            with open(args.config) as f:
                config = json.load(f)
            
            verifier = Stage2GCIVerifier(stage1_best, config, args.model, Path(args.geometry))
            
            # Skip CFD mode not applicable to GCI verification
            if args.skip_cfd:
                logger.warning("⚠️ --skip-cfd not supported for GCI verification (requires WSS data)")
                logger.info("💡 Use Stage 1 geometry-only optimization instead")
                return 1
            
            # Execute GCI verification
            logger.info(f"🔬 Using Stage 1 mesh: {stage1_best}")
            results = verifier.execute()
            
            if results["status"] == "SUCCESS":
                logger.info(f"✅ Stage 2 GCI verification completed")
                logger.info("📊 Final GCI Summary:")
                
                gci_results = results["gci_analysis"]
                logger.info(f"  Accepted mesh level: {results['accepted_level']}")
                logger.info(f"  GCI convergence: {gci_results['gci_21']:.1f}% (target: ≤{gci_results['tolerance_pct']}%)")
                logger.info(f"  Richardson apparent order: {gci_results['apparent_order']:.2f}")
                logger.info(f"  Refinement ratio: {gci_results['refinement_ratio']}")
                
                # WSS metrics summary
                wss_metrics = results["wss_metrics"]
                logger.info("  TAWSS (Time-Averaged Wall Shear Stress):")
                for level in ["coarse", "medium", "fine"]:
                    tawss = wss_metrics[level]["TAWSS"]["global_mean"]
                    logger.info(f"    {level}: {tawss:.3f} Pa")
                
                logger.info("  OSI (Oscillatory Shear Index):")
                for level in ["coarse", "medium", "fine"]:
                    osi = wss_metrics[level]["OSI"]["global_mean"]
                    logger.info(f"    {level}: {osi:.4f}")
                
                logger.info(f"  Mesh convergence: {'✅ CONVERGED' if gci_results['converged'] else '❌ NOT CONVERGED'}")
                logger.info(f"  Ready for production CFD studies: {results['accepted_mesh_path']}")
            else:
                logger.error(f"❌ Stage 2 GCI verification failed: {results.get('error', 'Unknown error')}")
                return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())