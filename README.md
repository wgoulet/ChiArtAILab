# ChiArtAILab
Project exploring use of Oracle OML with data from Chicago Art Institute's API

## Setup
Follow setup instructions to install and configure Oracle Machine Learning for Python client first: https://docs.oracle.com/en/database/oracle/machine-learning/oml4py/1/mlpug/oracle-machine-learning-python.html#GUID-D00976CA-3663-4F32-A6A2-B6BF5A843ADC

While the oml client is configured to use Oracle Wallet for authentication, credentials to connect to the OML DB are still required, set them in the OMLUSERNAME/OMLPASS environment variables prior to getting started.

## Overview
This project is mainly a learning exercise to become comfortable with machine learning concepts. I chose to use Oracle Cloud Infrastructure for this project as they have a seemingly generous free tier when using OML AutonomousDB.
My goal is to ingest data about artworks housed at the Chicago Art Institute, leveraging the great APIs they have available @ https://api.artic.edu/docs/#introduction.

