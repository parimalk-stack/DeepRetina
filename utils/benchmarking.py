# utils/benchmarking.py - Advanced Benchmarking and Comparison Module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    roc_curve, auc, roc_auc_score
)
from scipy import stats
import time
import logging
from collections import defaultdict, Counter
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class ModelBenchmark:
    """Comprehensive model benchmarking class"""
    
    def __init__(self):
        self.results = defaultdict(list)
        self.model_metadata = {}
        self.class_mapping = {
            'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3
        }
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
        
    def add_result(self, model_name, result):
        """Add a single result to the benchmark"""
        self.results[model_name].append(result)
        
    def add_batch_results(self, model_name, results):
        """Add batch results for a model"""
        self.results[model_name].extend(results)
        
    def set_model_metadata(self, model_name, metadata):
        """Set metadata for a model"""
        self.model_metadata[model_name] = metadata
        
    def calculate_metrics(self, model_name):
        """Calculate comprehensive metrics for a model"""
        if model_name not in self.results or not self.results[model_name]:
            return {}
            
        results = self.results[model_name]
        
        # Filter results with ground truth
        results_with_truth = [r for r in results if r.get('true_class')]
        
        if not results_with_truth:
            return {
                'total_predictions': len(results),
                'predictions_with_ground_truth': 0,
                'error': 'No ground truth available for evaluation'
            }
        
        # Extract predictions and ground truth
        y_true = [self.class_mapping[r['true_class']] for r in results_with_truth]
        y_pred = [self.class_mapping[r['predicted_class']] for r in results_with_truth]
        confidences = [r['confidence'] for r in results_with_truth]
        processing_times = [r.get('processing_time', 0) + r.get('inference_time', 0) for r in results_with_truth]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confidence statistics
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        # Correct vs incorrect confidence analysis
        correct_confidences = [c for i, c in enumerate(confidences) if y_true[i] == y_pred[i]]
        incorrect_confidences = [c for i, c in enumerate(confidences) if y_true[i] != y_pred[i]]
        
        # Processing time statistics
        avg_processing_time = np.mean(processing_times)
        processing_time_std = np.std(processing_times)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Class distribution
        class_distribution = Counter([self.reverse_class_mapping[y] for y in y_true])
        prediction_distribution = Counter([self.reverse_class_mapping[y] for y in y_pred])
        
        return {
            'total_predictions': len(results),
            'predictions_with_ground_truth': len(results_with_truth),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cohen_kappa': kappa,
            'per_class_metrics': {
                'precision': {self.reverse_class_mapping[i]: per_class_precision[i] for i in range(len(per_class_precision))},
                'recall': {self.reverse_class_mapping[i]: per_class_recall[i] for i in range(len(per_class_recall))},
                'f1_score': {self.reverse_class_mapping[i]: per_class_f1[i] for i in range(len(per_class_f1))}
            },
            'confidence_stats': {
                'mean': avg_confidence,
                'std': confidence_std,
                'correct_mean': np.mean(correct_confidences) if correct_confidences else 0,
                'incorrect_mean': np.mean(incorrect_confidences) if incorrect_confidences else 0
            },
            'processing_time_stats': {
                'mean': avg_processing_time,
                'std': processing_time_std,
                'min': np.min(processing_times),
                'max': np.max(processing_times)
            },
            'confusion_matrix': cm.tolist(),
            'class_distribution': dict(class_distribution),
            'prediction_distribution': dict(prediction_distribution)
        }
    
    def compare_models(self, model_names=None):
        """Compare multiple models"""
        if model_names is None:
            model_names = list(self.results.keys())
            
        comparison = {}
        all_metrics = {}
        
        # Calculate metrics for each model
        for model_name in model_names:
            metrics = self.calculate_metrics(model_name)
            all_metrics[model_name] = metrics
            
        # Model agreement analysis
        if len(model_names) >= 2:
            agreement_analysis = self._analyze_model_agreement(model_names)
            comparison['agreement_analysis'] = agreement_analysis
            
        # Performance comparison
        comparison['performance_summary'] = self._create_performance_summary(all_metrics)
        comparison['detailed_metrics'] = all_metrics
        
        return comparison
    
    def _analyze_model_agreement(self, model_names):
        """Analyze agreement between models"""
        try:
            # Get common samples across all models
            common_samples = None
            model_predictions = {}
            
            for model_name in model_names:
                results_with_truth = [r for r in self.results[model_name] if r.get('true_class')]
                filenames = set([r['filename'] for r in results_with_truth])
                
                if common_samples is None:
                    common_samples = filenames
                else:
                    common_samples = common_samples.intersection(filenames)
                    
                # Store predictions by filename
                model_predictions[model_name] = {
                    r['filename']: r['predicted_class'] for r in results_with_truth
                }
            
            if not common_samples:
                return {'error': 'No common samples found across models'}
            
            # Calculate pairwise agreement
            pairwise_agreement = {}
            agreement_matrix = np.zeros((len(model_names), len(model_names)))
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i == j:
                        agreement_matrix[i, j] = 1.0
                        continue
                    
                    agreements = 0
                    total = 0
                    
                    for filename in common_samples:
                        if filename in model_predictions[model1] and filename in model_predictions[model2]:
                            if model_predictions[model1][filename] == model_predictions[model2][filename]:
                                agreements += 1
                            total += 1
                    
                    agreement_rate = agreements / total if total > 0 else 0
                    agreement_matrix[i, j] = agreement_rate
                    pairwise_agreement[f"{model1}_vs_{model2}"] = {
                        'agreement_rate': agreement_rate,
                        'agreements': agreements,
                        'total_comparisons': total
                    }
            
            # Overall agreement (all models agree)
            unanimous_agreements = 0
            for filename in common_samples:
                predictions = [model_predictions[model][filename] for model in model_names 
                             if filename in model_predictions[model]]
                if len(set(predictions)) == 1:  # All predictions are the same
                    unanimous_agreements += 1
            
            unanimous_rate = unanimous_agreements / len(common_samples) if common_samples else 0
            
            return {
                'common_samples_count': len(common_samples),
                'pairwise_agreement': pairwise_agreement,
                'agreement_matrix': agreement_matrix.tolist(),
                'unanimous_agreement_rate': unanimous_rate,
                'unanimous_agreements': unanimous_agreements,
                'model_names': model_names
            }
            
        except Exception as e:
            logger.error(f"Error analyzing model agreement: {str(e)}")
            return {'error': str(e)}
    
    def _create_performance_summary(self, all_metrics):
        """Create performance summary table"""
        summary = {}
        
        # Extract key metrics for comparison
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'cohen_kappa']
        
        for metric in metrics_to_compare:
            summary[metric] = {}
            values = []
            
            for model_name, metrics in all_metrics.items():
                if metric in metrics and not isinstance(metrics[metric], str):
                    summary[metric][model_name] = metrics[metric]
                    values.append(metrics[metric])
                else:
                    summary[metric][model_name] = 0.0
            
            # Add ranking
            if values:
                sorted_models = sorted(all_metrics.keys(), 
                                     key=lambda x: summary[metric].get(x, 0), reverse=True)
                for i, model in enumerate(sorted_models):
                    if f"{metric}_rank" not in summary:
                        summary[f"{metric}_rank"] = {}
                    summary[f"{metric}_rank"][model] = i + 1
        
        # Processing time comparison
        summary['avg_processing_time'] = {}
        for model_name, metrics in all_metrics.items():
            time_stats = metrics.get('processing_time_stats', {})
            summary['avg_processing_time'][model_name] = time_stats.get('mean', 0)
        
        # Confidence comparison
        summary['avg_confidence'] = {}
        for model_name, metrics in all_metrics.items():
            conf_stats = metrics.get('confidence_stats', {})
            summary['avg_confidence'][model_name] = conf_stats.get('mean', 0)
        
        return summary

def create_model_benchmark():
    """Factory function to create a new ModelBenchmark instance"""
    return ModelBenchmark()

def quick_model_comparison(results_dict, output_dir=None):
    """Quick comparison of multiple model results"""
    benchmark = ModelBenchmark()
    
    # Add results for each model
    for model_name, results in results_dict.items():
        benchmark.add_batch_results(model_name, results)
    
    # Generate comparison
    comparison = benchmark.compare_models()
    
    return comparison

def calculate_statistical_significance(results1, results2, metric='accuracy'):
    """Calculate statistical significance between two model results"""
    try:
        # Extract metric values with ground truth
        values1 = []
        values2 = []
        
        # Create filename mapping for paired comparison
        results1_dict = {r['filename']: r for r in results1 if r.get('true_class')}
        results2_dict = {r['filename']: r for r in results2 if r.get('true_class')}
        
        common_files = set(results1_dict.keys()).intersection(set(results2_dict.keys()))
        
        class_mapping = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3}
        
        for filename in common_files:
            r1 = results1_dict[filename]
            r2 = results2_dict[filename]
            
            if metric == 'accuracy':
                # Binary accuracy for each sample
                acc1 = 1 if r1['predicted_class'] == r1['true_class'] else 0
                acc2 = 1 if r2['predicted_class'] == r2['true_class'] else 0
                values1.append(acc1)
                values2.append(acc2)
            elif metric == 'confidence':
                values1.append(r1['confidence'])
                values2.append(r2['confidence'])
        
        if len(values1) < 2:
            return {'error': 'Insufficient paired samples for statistical test'}
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(values1, values2)
        
        # Effect size (Cohen's d)
        diff = np.array(values1) - np.array(values2)
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        
        return {
            'metric': metric,
            'n_samples': len(values1),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'effect_size_interpretation': (
                'Large' if abs(cohens_d) >= 0.8 else
                'Medium' if abs(cohens_d) >= 0.5 else
                'Small' if abs(cohens_d) >= 0.2 else
                'Negligible'
            )
        }
        
    except Exception as e:
        logger.error(f"Error calculating statistical significance: {str(e)}")
        return {'error': str(e)}
