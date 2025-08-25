"""
Unit tests for QualityAnalyzer module.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from mesh_optim.stage1.quality_analyzer import QualityAnalyzer
from mesh_optim.stage1.config_manager import ConfigManager


class TestQualityAnalyzer:
    """Test suite for QualityAnalyzer"""
    
    def test_initialization(self, sample_config_file):
        """Test QualityAnalyzer initialization"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        assert analyzer.config is not None
        assert analyzer.targets is not None
        assert analyzer.constants is not None
        assert analyzer.quality_history == []
        assert analyzer.current_iteration == 0
    
    @patch('mesh_optim.stage1.quality_analyzer.run_command')
    @patch('mesh_optim.stage1.quality_analyzer.check_mesh_quality')
    def test_serial_quality_check(self, mock_check_mesh, mock_run_command, sample_config_file, temp_dir):
        """Test serial mesh quality assessment"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        # Mock mesh quality results
        mock_check_mesh.return_value = {
            "maxNonOrtho": 45.0,
            "maxSkewness": 2.5,
            "cells": 50000,
            "meshOK": True
        }
        
        result = analyzer._serial_quality_check(temp_dir, "test_phase", "wall_aorta")
        
        assert result["maxNonOrtho"] == 45.0
        assert result["maxSkewness"] == 2.5
        assert result["cells"] == 50000
        assert result["meshOK"] is True
        mock_check_mesh.assert_called_once()
    
    def test_parallel_checkmesh_parsing(self, sample_config_file):
        """Test parsing of parallel checkMesh output"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        # Sample parallel checkMesh output
        output = """
        Checking mesh...
        
        cells:  50000
        faces:  150000
        points: 75000
        
        Max non-orthogonality = 42.5
        Max skewness = 2.1
        Max aspect ratio = 45.2e+01
        
        Mesh OK.
        """
        
        metrics = analyzer._parse_parallel_checkmesh(output)
        
        assert metrics["cells"] == 50000
        assert metrics["faces"] == 150000
        assert metrics["points"] == 75000
        assert metrics["maxNonOrtho"] == 42.5
        assert metrics["maxSkewness"] == 2.1
        assert metrics["maxAspectRatio"] == 45.2e+01
        assert metrics["meshOK"] is True
    
    def test_quality_history_tracking(self, sample_config_file):
        """Test quality history tracking and limits"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        # Add multiple quality entries
        for i in range(15):  # More than the 10-entry limit
            metrics = {
                "maxNonOrtho": 40.0 + i,
                "maxSkewness": 2.0 + i * 0.1,
                "cells": 50000 + i * 1000,
                "meshOK": True,
                "phase": f"test_{i}"
            }
            analyzer._update_quality_history(metrics)
        
        # Should keep only last 10 entries
        assert len(analyzer.quality_history) == 10
        assert analyzer.quality_history[-1]["iteration"] == 15
        assert analyzer.quality_history[0]["iteration"] == 6  # Oldest kept entry
    
    def test_convergence_detection(self, sample_config_file):
        """Test convergence detection algorithm"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        # Add converged quality history (stable values)
        stable_values = [
            {"maxNonOrtho": 40.0, "maxSkewness": 2.0, "cells": 50000},
            {"maxNonOrtho": 40.1, "maxSkewness": 2.0, "cells": 50100},
            {"maxNonOrtho": 39.9, "maxSkewness": 2.1, "cells": 49900}
        ]
        
        for i, values in enumerate(stable_values, 1):
            values.update({"meshOK": True, "phase": f"test_{i}"})
            analyzer._update_quality_history(values)
        
        converged, reason = analyzer.check_convergence()
        
        # Should detect convergence due to low coefficient of variation
        assert converged is True
        assert "converged" in reason.lower()
    
    def test_non_convergence_detection(self, sample_config_file):
        """Test detection of non-convergence"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        # Add oscillating quality history
        oscillating_values = [
            {"maxNonOrtho": 30.0, "maxSkewness": 1.5, "cells": 40000},
            {"maxNonOrtho": 50.0, "maxSkewness": 3.0, "cells": 60000},
            {"maxNonOrtho": 35.0, "maxSkewness": 2.0, "cells": 45000}
        ]
        
        for i, values in enumerate(oscillating_values, 1):
            values.update({"meshOK": True, "phase": f"test_{i}"})
            analyzer._update_quality_history(values)
        
        converged, reason = analyzer.check_convergence()
        
        # Should not detect convergence due to high variation
        assert converged is False
        assert "not converged" in reason.lower()
    
    def test_quality_constraints_checking(self, sample_config_file):
        """Test quality constraints validation"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        # Good quality metrics
        snap_metrics = {"maxNonOrtho": 40.0, "maxSkewness": 2.0, "meshOK": True}
        layer_metrics = {"maxNonOrtho": 45.0, "maxSkewness": 2.5, "meshOK": True}
        layer_coverage = {"coverage_overall": 0.8}
        
        meets_constraints = analyzer.meets_quality_constraints(
            snap_metrics, layer_metrics, layer_coverage
        )
        
        assert meets_constraints is True
        
        # Poor quality metrics
        poor_layer_metrics = {"maxNonOrtho": 80.0, "maxSkewness": 6.0, "meshOK": False}
        poor_coverage = {"coverage_overall": 0.3}
        
        meets_constraints_poor = analyzer.meets_quality_constraints(
            snap_metrics, poor_layer_metrics, poor_coverage
        )
        
        assert meets_constraints_poor is False
    
    def test_layer_quality_assessment(self, sample_config_file, temp_dir):
        """Test layer quality assessment"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        # Mock layer coverage parsing
        with patch('mesh_optim.stage1.quality_analyzer.parse_layer_coverage') as mock_parse:
            mock_parse.return_value = {
                "coverage_overall": 0.85,
                "perPatch": {"wall_aorta": 0.85, "inlet": 0.90}
            }
            
            result = analyzer.assess_layer_quality(temp_dir)
            
            assert result["coverage_overall"] == 0.85
            assert result["quality_assessment"]["coverage_good"] is True
            assert result["quality_assessment"]["meets_target"] is True
    
    def test_quality_diagnosis(self, sample_config_file):
        """Test quality issue diagnosis"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        # Test quality degradation from snap to layers
        snap_metrics = {"maxNonOrtho": 35.0, "maxSkewness": 2.0}
        layer_metrics = {"maxNonOrtho": 55.0, "maxSkewness": 3.5}  # Degraded
        
        diagnosis = analyzer.diagnose_quality_issues(snap_metrics, layer_metrics)
        
        assert diagnosis["delta_nonortho"] == 20.0
        assert diagnosis["delta_skewness"] == 1.5
        assert diagnosis["layers_degrade_quality"] is True
        assert len(diagnosis["suggestions"]) > 0
        assert "reduce firstLayerThickness" in diagnosis["suggestions"][0].lower()
    
    def test_coefficient_of_variation(self, sample_config_file):
        """Test coefficient of variation calculation"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        # Test stable values (low CV)
        stable_values = [10.0, 10.1, 9.9, 10.0]
        cv_stable = analyzer._coefficient_of_variation(stable_values)
        assert cv_stable < 0.05  # Should be very low
        
        # Test variable values (high CV)
        variable_values = [5.0, 15.0, 8.0, 12.0]
        cv_variable = analyzer._coefficient_of_variation(variable_values)
        assert cv_variable > 0.1  # Should be higher
        
        # Test edge cases
        assert analyzer._coefficient_of_variation([0, 0, 0]) == 0.0
        assert analyzer._coefficient_of_variation([5.0]) == 0.0
    
    def test_fallback_metrics(self, sample_config_file):
        """Test fallback metrics generation"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        fallback = analyzer._get_fallback_metrics("test_phase")
        
        assert fallback["maxNonOrtho"] == 999
        assert fallback["maxSkewness"] == 999
        assert fallback["meshOK"] is False
        assert fallback["phase"] == "test_phase"
        assert fallback["error"] is True
    
    def test_quality_report_export(self, sample_config_file, temp_dir):
        """Test quality analysis report export"""
        config_manager = ConfigManager(str(sample_config_file))
        analyzer = QualityAnalyzer(config_manager)
        
        # Add some quality history
        for i in range(3):
            metrics = {
                "maxNonOrtho": 40.0 + i,
                "maxSkewness": 2.0 + i * 0.1,
                "cells": 50000,
                "meshOK": True,
                "phase": f"test_{i}"
            }
            analyzer._update_quality_history(metrics)
        
        # Export report
        report_path = temp_dir / "quality_report.json"
        analyzer.export_quality_report(report_path)
        
        assert report_path.exists()
        
        # Verify report content
        import json
        with open(report_path) as f:
            report = json.load(f)
        
        assert "quality_targets" in report
        assert "quality_history" in report
        assert "total_iterations" in report
        assert len(report["quality_history"]) == 3