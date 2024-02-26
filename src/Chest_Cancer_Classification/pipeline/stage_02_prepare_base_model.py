from Chest_Cancer_Classification.config.configuraton import ConfigurationManager
from Chest_Cancer_Classification.components.prepare_base_model import PrepareBaseModel
from Chest_Cancer_Classification import logger

STAGE_NAME = "Prepare Base Model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self) :
        pass
    
    def main(self):
        config = ConfigurationManager()
        preapre_base_model_config= config.get_prepare_base_model_config()
        preapre_base_model = PrepareBaseModel(config=preapre_base_model_config)
        preapre_base_model.get_base_model()
        preapre_base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f"****************")
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e 