#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info("Downloading " + args.input_artifact + " from W&B")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    logger.info("Reading " + artifact_path)
    df = pd.read_csv(artifact_path)
    
    # cleaning 
    logger.info("A little clean up")
    
    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    logger.info("clamped price column")
    
    df = df[idx].copy()
    
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info("converted last_review to datetime")
    
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    logger.info("applied longitude/latitude fix")

    filename = "processed_data.csv"
    df.to_csv(filename, index=False)
    
    logger.info("Storing cleaned sample.csv in W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)
    run.log_artifact(artifact)
    
    os.remove(filename)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name for the input artifact (sample.csv)",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact (clean_sample.csv)",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="A descriptive text for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="To restrict outliers for price: minimal price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="To restrict outliers for price: maximal price",
        required=True
    )


    args = parser.parse_args()

    go(args)
