from experiments.t_dm.operator import DiscoMultiOperator

class DiscoMultiOperator_lr(DiscoMultiOperator):
    def _get_optuna_fn(self, train_params):
        from utils.train_ops import train, get_optuna_params
        optuna_params = get_optuna_params(train_params)

        def obj_fn(trial):
            def spec_update_fn(specs, trial):
                self._train_materials = None, self._train_materials[1]
                lr = specs['train']['learning_rate']
                specs['train']['learning_rate'] = lr = trial.suggest_loguniform('learning_rate', 1e-6, lr)
                return f'lr={lr:.1e}'

            self._mode_trees = [], [] # force init
            self.setup_optuna_mode(spec_update_fn, trial)

            return train(optuna_params, self)['key']
        return obj_fn