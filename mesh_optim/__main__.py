"""
CLI entry point for mesh optimization package
"""

import argparse
import sys
import logging
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

  # Stage 2: QoI-driven optimization for RANS
  python -m mesh_optim stage2 --geometry cases_input/patient1 --model RANS

  # Stage 2: QoI-driven optimization for LES
  python -m mesh_optim stage2 --geometry cases_input/patient1 --model LES --config mesh_optim/configs/stage2_les.json
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
    stage2_parser = subparsers.add_parser('stage2', help='QoI-driven mesh optimization')
    stage2_parser.add_argument('--geometry', required=True, help='Path to geometry directory')
    stage2_parser.add_argument('--model', choices=['LAMINAR', 'RANS', 'LES'], default='RANS', help='Flow model')
    stage2_parser.add_argument('--config', help='Configuration file (default: auto-select based on model)')
    stage2_parser.add_argument('--output', help='Output directory (default: output/<patient>/meshOptimizer/stage2_<model>)') 
    stage2_parser.add_argument('--max-iterations', type=int, default=3, help='Maximum QoI iterations')
    stage2_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    stage2_parser.add_argument('--skip-cfd', action='store_true', help='Skip CFD simulations (mesh-only mode)')
    
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
            from .stage2_qoi import Stage2QOIOptimizer
            
            # Generate default output path if not specified
            if not args.output:
                patient_name = Path(args.geometry).name
                model_suffix = args.model.lower()
                args.output = Path("output") / patient_name / "meshOptimizer" / f"stage2_{model_suffix}"
            
            cfd_mode = "mesh-only" if args.skip_cfd else "with CFD"
            logger.info(f"Starting Stage 2 QoI optimization: {args.geometry} ({args.model}) [{cfd_mode}]")
            logger.info(f"Output directory: {args.output}")
            optimizer = Stage2QOIOptimizer(args.geometry, args.model, args.config, args.output)
            
            # Disable CFD if requested
            if args.skip_cfd:
                optimizer.cfd_solver = None
                logger.info("⚠️ CFD simulations disabled - using mesh-based estimates only")
            
            if args.max_iterations:
                optimizer.max_iterations = args.max_iterations
                
            result_dir, qoi_metrics = optimizer.iterate_until_converged()
            
            if qoi_metrics.get("valid", False):
                logger.info(f"✅ Stage 2 completed: {result_dir}")
                logger.info("📊 Final QoI Summary:")
                
                # Mesh quality summary
                mesh_quality = qoi_metrics.get("mesh_quality", {})
                logger.info(f"  Cells: {mesh_quality.get('cells', 0):,}")
                logger.info(f"  Max skewness: {mesh_quality.get('max_skewness', 0):.2f}")
                logger.info(f"  Max non-orthogonality: {mesh_quality.get('max_nonortho', 0):.1f}°")
                
                # Layer analysis summary
                layer_analysis = qoi_metrics.get("layer_analysis", {})
                logger.info(f"  Layer coverage: {layer_analysis.get('coverage_overall', 0)*100:.1f}%")
                
                # CFD results summary (if available)
                cfd_results = qoi_metrics.get("cfd_results", {})
                if cfd_results and not cfd_results.get("error"):
                    logger.info("  CFD Analysis:")
                    yplus = cfd_results.get("yplus_analysis", {})
                    if yplus.get("available"):
                        logger.info(f"    Y+ range: {yplus.get('min_yplus', 0):.1f}-{yplus.get('max_yplus', 0):.1f} (mean: {yplus.get('mean_yplus', 0):.1f})")
                        logger.info(f"    Y+ coverage: {yplus.get('coverage_in_range', 0)*100:.1f}%")
                    
                    wss = cfd_results.get("wss_analysis", {})
                    if wss.get("available"):
                        logger.info(f"    WSS range: {wss.get('min_wss', 0):.1f}-{wss.get('max_wss', 0):.1f} Pa (mean: {wss.get('mean_wss', 0):.1f} Pa)")
                
                logger.info(f"  Overall converged: {qoi_metrics.get('converged', False)}")
            else:
                logger.warning(f"⚠️ Stage 2 incomplete: {result_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())