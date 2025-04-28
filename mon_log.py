import logging
import random
import time
import mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' )
logger = logging.getLogger(__name__)

def generate_random_data():

    try:
        mlflow.start_run()
        mlflow.log_params({'app_name':'MyApp', 'version':1.0})
        start_time = time.time()



        while True:
            data = random.randint(1, 100)
            logging.info("Starting Monitoring process!")

            logger.info(f"Generating data: {data}")

            mlflow.log_metric("my_metric", data)

            uptime =  time.time() - start_time

            mlflow.log_metric(f"app_uptime_seconds", uptime)
            time.sleep(2)

    except KeyboardInterrupt:
        logging.info("Keyboard Interruption detected. Monitoring & Logging terminated.")
    except Exception as e:
        logging.error(f"Monitoring error detected: {str(e)}")
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    try:
        generate_random_data()
    except Exception as e:
        logging.error("An error occured!")
    finally:
        logging.info("Monitoring & Logging has stopped.")