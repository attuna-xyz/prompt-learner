"""Class for grid search optimization."""
from prompt_learner.optimizers.optimizer import Optimizer
from prompt_learner.evals.metrics.accuracy import Accuracy
from tqdm import tqdm

class GridSearch(Optimizer):
    """Grid search optimizer."""
    def search(self, param_grid: dict):
        """Search for the best hyperparameters given in the 
        parameter grid."""
        best_score = 0
        best_params = {}
        all_results = []
        all_samplers = param_grid.get('sampler',None)
        all_templates = param_grid.get('template', [self.prompt.template])
        all_adapters = param_grid.get('adapter',None)
        if all_samplers is not None:
            return self.search_samplers(all_samplers, all_templates, all_adapters)
        total_iterations = len(all_templates) * len(all_adapters)
        with tqdm(total=total_iterations, desc="Grid Search Progress") as pbar:
            for template in all_templates:
                self.prompt.translate(template)
                for adapter in all_adapters:
                    acc, _ = Accuracy(self.prompt.template.task).compute(self.prompt, adapter)
                    score = acc
                    curr_params = {'score': score, 
                                'template': template.class_repr(),
                                'adapter': repr(adapter)}
                    all_results.append(curr_params)
                    if score > best_score:
                        best_score = score
                        best_params = curr_params
                    pbar.update(1)

        return best_params, all_results
    
    def search_samplers(self, all_samplers, all_templates, all_adapters):
        """Search when samplers are in param grid"""
        best_score = 0
        best_params = {}
        all_results = []
        total_iterations = len(all_samplers) * len(all_templates) * len(all_adapters)
        with tqdm(total=total_iterations, desc="Grid Search Progress") as pbar:
            for sampler in all_samplers:
                sampler.task = self.prompt.template.task
                sampler.select_examples()
                for template in all_templates:
                    self.prompt.translate(template)
                    for adapter in all_adapters:
                        acc, _ = Accuracy(sampler.task).compute(self.prompt, adapter)
                        score = acc
                        curr_params = {'score': score, 'sampler': repr(sampler),
                                       'template': template.class_repr(),
                                       'adapter': repr(adapter)}
                        all_results.append(curr_params)
                        if score > best_score:
                            best_score = score
                            best_params = curr_params
                        pbar.update(1)

        return best_params, all_results

    def compute_metrics(self):
        """Compute metrics for a given prompt and adapter on a task."""
        pass