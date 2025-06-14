from urllib.request import urlretrieve
from celery import shared_task
import subprocess
import logging

logger = logging.getLogger(__name__)

@shared_task
def fetch_modis_data():
    """
    Fetches MODIS Aqua SST Level-3 Mapped NRT data using the NASA OBPG API.
    """
    try:
        # The curl command to fetch the data
        curl_command = [
            'curl',
            '-d',
            "results_as_file=1&sensor_id=7&dtid=1061&backdays=3&datetype=1&subType=1&addurl=1&prod_id=sst&resolution_id=4km",
            'https://oceandata.sci.gsfc.nasa.gov/api/file_search'
        ]

        logger.info(f"Executing curl command: {' '.join(curl_command)}")

        # Execute the curl command
        process = subprocess.run(
            curl_command,
            capture_output=True,
            text=True,
            check=True  # Raise an exception if the command fails
        )

        # The output of the curl command will be the list of file URLs
        file_list = process.stdout.strip().split('\n')

        logger.info("Successfully fetched MODIS data file list:")
        for file_url in file_list:
            logger.info(file_url)

        # You can add further processing here, e.g., downloading the files
        # For example:
        for file_url in file_list:
            download_file.delay(file_url) # Assuming you have another task for downloading

        return "MODIS data file list fetched successfully."

    except subprocess.CalledProcessError as e:
        logger.error(f"Curl command failed: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return f"Error fetching MODIS data: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}"

# Example of a potential task to download files (optional)
from urllib.request import urlretrieve

@shared_task
def download_file(url):
    try:
        filename = url.split('/')[-1]
        urlretrieve(url, filename)
        logger.info(f"Downloaded: {filename}")
        return f"Downloaded: {filename}"
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return f"Error downloading {url}: {e}"


# from urllib.request import urlretrieve
# from celery import shared_task
# import subprocess
# import logging

# logger = logging.getLogger(__name__)

# @shared_task
# def fetch_modis_data():
#     """
#     Fetches MODIS Aqua SST and Chlorophyll Level-3 Mapped NRT data using the NASA OBPG API.
#     """
#     try:
#         base_url = 'https://oceandata.sci.gsfc.nasa.gov/api/file_search'
#         payloads = [
#             {
#                 "name": "SST",
#                 "data": "results_as_file=1&sensor_id=7&dtid=1061&backdays=3&datetype=1&subType=1&addurl=1&prod_id=sst&resolution_id=4km"
#             },
#             {
#                 "name": "Chlorophyll",
#                 "data": "results_as_file=1&sensor_id=7&dtid=1004&backdays=3&datetype=1&subType=1&addurl=1&prod_id=chlor_a&resolution_id=4km"
#             }

#         ]

#         all_files = []

#         for payload in payloads:
#             curl_command = [
#                 'curl',
#                 '-d',
#                 payload["data"],
#                 base_url
#             ]

#             logger.info(f"Executing curl for {payload['name']}: {' '.join(curl_command)}")

#             process = subprocess.run(
#                 curl_command,
#                 capture_output=True,
#                 text=True,
#                 check=True
#             )

#             file_list = process.stdout.strip().split('\n')
#             logger.info(f"{payload['name']} file list fetched:")
#             for file_url in file_list:
#                 logger.info(file_url)
#                 all_files.append(file_url)

#         # Trigger download for all files found
#         for file_url in all_files:
#             download_file.delay(file_url)

#         return f"Successfully fetched and dispatched download tasks for {len(all_files)} MODIS files."

#     except subprocess.CalledProcessError as e:
#         logger.error(f"Curl command failed: {e}")
#         logger.error(f"Stderr: {e.stderr}")
#         return f"Error fetching MODIS data: {e}"
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         return f"Unexpected error: {e}"


# @shared_task
# def download_file(url):
#     try:
#         filename = url.split('/')[-1]
#         urlretrieve(url, filename)
#         logger.info(f"Downloaded: {filename}")
#         return f"Downloaded: {filename}"
#     except Exception as e:
#         logger.error(f"Error downloading {url}: {e}")
#         return f"Error downloading {url}: {e}"

