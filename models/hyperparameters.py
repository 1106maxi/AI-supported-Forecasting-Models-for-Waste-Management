class XGBoostParams:
    """XGBoost hyperparameters for different companies."""
    
    @staticmethod
    def get_all_params():
        """Returns dictionary of best XGBoost parameters for all companies."""
        return {
            "RegionalWaste Management": {
                "colsample_bytree": 0.8,
                "gamma": 0,
                "learning_rate": 0.09,
                "max_depth": 3,
                "min_child_weight": 3,
                "n_estimators": 80,
                "reg_alpha": 0.5
            },
            "GreenWaste Solutions": {
                "colsample_bytree": 0.8,
                "gamma": 0,
                "learning_rate": 0.13,
                "max_depth": 6,
                "min_child_weight": 1,
                "n_estimators": 70,
                "reg_alpha": 0.1
            },
            "IndustrialProcess Ltd": {
                "colsample_bytree": 1.0,
                "gamma": 0,
                "learning_rate": 0.07,
                "max_depth": 4,
                "min_child_weight": 5,
                "n_estimators": 70,
                "reg_alpha": 0
            },
            "BuildRight Construction": {
                "colsample_bytree": 0.8,
                "gamma": 0,
                "learning_rate": 0.11,
                "max_depth": 3,
                "min_child_weight": 1,
                "n_estimators": 90,
                "reg_alpha": 0.5
            },
            "CommercialServices Inc": {
                "colsample_bytree": 0.8,
                "gamma": 0.3,
                "learning_rate": 0.03,
                "max_depth": 5,
                "min_child_weight": 5,
                "n_estimators": 130,
                "reg_alpha": 0.1
            },
            "MunicipalWaste Co": {
                "colsample_bytree": 0.8,
                "gamma": 0,
                "learning_rate": 0.05,
                "max_depth": 4,
                "min_child_weight": 5,
                "n_estimators": 100,
                "reg_alpha": 0.5
            }
        }
    
    @staticmethod
    def get_company_params(company_name):
        """Convenience method to get parameters for a specific company."""
        all_params = XGBoostParams.get_all_params()
        if company_name in all_params:
            return all_params[company_name]
        else:
            raise ValueError(f"No parameters found for company: {company_name}")

class CatBoostParams:
    """CatBoost hyperparameters for different companies.
    
    NOTE: These parameters are translated from XGBoost parameters and are for test purposes only.
    A proper grid search for CatBoost has not been performed, so these may not be optimal.
    """
    
    @staticmethod
    def get_all_params():
        """Returns dictionary of CatBoost parameters for all companies (translated from XGBoost).
        
        These are approximate translations and should be validated with proper hyperparameter tuning.
        """
        return {
            "RegionalWaste Management": {
                "rsm": 0.8,                # Translated from colsample_bytree
                "learning_rate": 0.09,
                "depth": 3,                # Translated from max_depth
                "min_data_in_leaf": 10,    # Approximate from min_child_weight
                "iterations": 80,          # Translated from n_estimators
                "l1_leaf_reg": 0.5,        # Translated from reg_alpha
                "verbose": False
            },
            "GreenWaste Solutions": {
                "rsm": 0.8,
                "learning_rate": 0.13,
                "depth": 6,
                "min_data_in_leaf": 5,     # Approximate from min_child_weight
                "iterations": 70,
                "l1_leaf_reg": 0.1,
                "verbose": False
            },
            "IndustrialProcess Ltd": {
                "rsm": 1.0,
                "learning_rate": 0.07,
                "depth": 4,
                "min_data_in_leaf": 15,    # Approximate from min_child_weight
                "iterations": 70,
                "l1_leaf_reg": 0,
                "verbose": False
            },
            "BuildRight Construction": {
                "rsm": 0.8,
                "learning_rate": 0.11,
                "depth": 3,
                "min_data_in_leaf": 5,     # Approximate from min_child_weight
                "iterations": 90,
                "l1_leaf_reg": 0.5,
                "verbose": False
            },
            "CommercialServices Inc": {
                "rsm": 0.8,
                "learning_rate": 0.03,
                "depth": 5,
                "min_data_in_leaf": 15,    # Approximate from min_child_weight
                "iterations": 130,
                "l1_leaf_reg": 0.1,
                "l2_leaf_reg": 0.3,        # Approximated from gamma
                "verbose": False
            },
            "MunicipalWaste Co": {
                "rsm": 0.8,
                "learning_rate": 0.05,
                "depth": 4,
                "min_data_in_leaf": 15,    # Approximate from min_child_weight
                "iterations": 100,
                "l1_leaf_reg": 0.5,
                "verbose": False
            }
        }
    
    @staticmethod
    def get_company_params(company_name):
        """Convenience method to get parameters for a specific company."""
        all_params = CatBoostParams.get_all_params()
        if company_name in all_params:
            return all_params[company_name]
        else:
            raise ValueError(f"No parameters found for company: {company_name}")