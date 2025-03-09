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