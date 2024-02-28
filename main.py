from Chest_Cancer_Classification import logger
from Chest_Cancer_Classification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Chest_Cancer_Classification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Chest_Cancer_Classification.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from Chest_Cancer_Classification.pipeline.stage_04_model_evaluation import ModelEvalutionPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx======x")
except Exception as e:
    logger.exception(e)
    raise e 



STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f"*****************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx======x")
except Exception as e:
    logger.exception(e)
    raise e 


STAGE_NAME = "Model Training"

try:
    logger.info(f"*****************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx======x")
except Exception as e:
    logger.exception(e)
    raise e 


STAGE_NAME = "Model Evalution Stage"

try:
    logger.info(f"*****************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<")
    obj = ModelEvalutionPipeline()
    obj.main()
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx======x")
except Exception as e:
    logger.exception(e)
    raise e 
