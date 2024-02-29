from Chest_Cancer_Classification.config.configuraton import ConfigurationManager
from Chest_Cancer_Classification.components.model_evalution_mlflow import Evaluation
from Chest_Cancer_Classification import logger


STAGE_NAME = "Model Evalution"

class ModelEvalutionPipeline:
    def __init__(self) :
        pass

    def main(self):
            config = ConfigurationManager()
            eval_config = config.get_evaluation_config()
            evaluation = Evaluation(eval_config)
            evaluation.evaluation()
            evaluation.save_score()
            # evaluation.log_into_mlflow()    


if __name__ == '__main__':
    try:
        logger.info(f"****************")
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<")
        obj = ModelEvalutionPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e 
