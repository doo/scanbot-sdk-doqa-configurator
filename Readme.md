# Scanbot SDK Document Quality Analyzer (DoQA) Configurator

## Introduction

This tool allows you to create a customized config for the Document Quality Analyzer from the Scanbot SDK.

The goal of the Document Quality Analyzer is to decide if a user-provided image of a document is of good enough quality to proceed, or if the user should be asked to provide an image of better quality.
This distinction between good and bad quality images is difficult and largely depends on the use-case:

- What type of documents will be scanned? E.g. receipts, invoices, contracts
- What part of the document carries the important information? E.g. is it important that the fine-print on the image is readable? Or is it sufficient if the larger text is readable?
- What are the capabilities of the next processing passes? E.g. text with poor contrast might be fine for OCR but is difficult to read for humans.

By providing examples of images that have sufficient or insufficient quality for your use-case, this tool will create a fine-tuned configuration file to optimize the DoQA performance for you.

## Prerequisites

To run this tool, you need to have the following:

- [Docker](https://docs.docker.com/engine/install/) with [Docker Compose](https://docs.docker.com/compose/install/) support
- Images of documents that are of sufficient (good) or insufficient (bad) quality for your use-case.
  Per class good / bad you should provide at least 100 samples (200 in total). The more samples you provide, the better final configuration will fit to your use-case.
- A special license key that is only valid for the DoQA Configurator. This license key will be different from the license key you use in your app. Please contact [customer support](https://docs.scanbot.io/support/) to obtain it.
- You need to know the version of the ScanbotSDK Core used in your release. Please also contact [customer support](https://docs.scanbot.io/support/) to obtain it.

## Usage

- Clone or download the repository
  ```
  git clone https://github.com/doo/scanbot-sdk-doqa-configurator
  ```
- Place the training images into the folders `data/bad` & `data/good`.
  Images should be in JPG or PNG format.
- Put the version of the ScanbotSDK Core inside the `docker-compose.yaml`.
- Run the following command to produce the custom configuration:
  ```
  docker compose run --build --rm sbsdk-doqa-configurator --scanbotsdk_license_key=YOUR_LICENSE_KEY
  ```
- Your config will be created in `data/DoQA_config.txt`. Please provide the contents of this file during the configuration of the Scanbot SDK.
