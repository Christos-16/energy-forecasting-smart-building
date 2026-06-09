#!/usr/bin/env python3
"""
Extract individual sensors from IoT dataset to ThesisML format.

Reads the large measurements CSV and extracts specific sensors
to individual CSV files with [timestamp, value] format.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np


def extract_sensor(
    input_csv: Path,
    sensor_name: str,
    output_csv: Path,
    chunk_size: int = 100000
):
    """Extract a single sensor from the dataset."""

    print(f"\n📦 Extracting sensor: {sensor_name}")
    print(f"   Input: {input_csv}")
    print(f"   Output: {output_csv}")

    # Initialize accumulator
    sensor_data = []
    total_rows = 0
    matched_rows = 0

    # Stream through chunks
    chunk_iter = pd.read_csv(input_csv, chunksize=chunk_size)

    for chunk_num, chunk in enumerate(chunk_iter, 1):
        total_rows += len(chunk)

        # Filter for this sensor
        sensor_chunk = chunk[chunk['item'] == sensor_name].copy()

        if len(sensor_chunk) > 0:
            matched_rows += len(sensor_chunk)

            # Keep only timestamp and value
            sensor_chunk = sensor_chunk[['ts', 'value']].copy()

            # Rename for ThesisML compatibility
            sensor_chunk.columns = ['timestamp', 'value']

            sensor_data.append(sensor_chunk)

        # Progress
        if chunk_num % 50 == 0:
            print(f"   📊 Processed {total_rows:,} rows, found {matched_rows:,} measurements")

    if not sensor_data:
        print(f"   ⚠️  No data found for sensor: {sensor_name}")
        return None

    # Combine all chunks
    print(f"   🔗 Combining {len(sensor_data)} chunks...")
    df = pd.concat(sensor_data, ignore_index=True)

    # Parse timestamps
    print(f"   ⏰ Parsing timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    # Remove nulls
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before > after:
        print(f"   🧹 Removed {before - after:,} rows with null values")

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    after = len(df)
    if before > after:
        print(f"   🧹 Removed {before - after:,} duplicate timestamps")

    # Ensure numeric value
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna()

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Stats
    print(f"   ✅ Saved {len(df):,} measurements")
    print(f"   📅 Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   📊 Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")

    return df


def extract_multiple_sensors(
    input_csv: Path,
    sensor_names: List[str],
    output_dir: Path,
    chunk_size: int = 100000
):
    """Extract multiple sensors efficiently (single pass through data)."""

    print(f"\n{'='*80}")
    print(f"🚀 Extracting {len(sensor_names)} sensors")
    print(f"{'='*80}")

    # Initialize accumulators for all sensors
    sensor_data = {sensor: [] for sensor in sensor_names}
    sensor_counts = {sensor: 0 for sensor in sensor_names}

    total_rows = 0

    # Stream through chunks
    print(f"\n📖 Reading {input_csv}...")
    chunk_iter = pd.read_csv(input_csv, chunksize=chunk_size)

    for chunk_num, chunk in enumerate(chunk_iter, 1):
        total_rows += len(chunk)

        # Filter for each sensor
        for sensor in sensor_names:
            sensor_chunk = chunk[chunk['item'] == sensor].copy()

            if len(sensor_chunk) > 0:
                sensor_counts[sensor] += len(sensor_chunk)

                # Keep only timestamp and value
                sensor_chunk = sensor_chunk[['ts', 'value']].copy()
                sensor_chunk.columns = ['timestamp', 'value']

                sensor_data[sensor].append(sensor_chunk)

        # Progress
        if chunk_num % 50 == 0:
            print(f"   📊 Processed {total_rows:,} rows")
            for sensor in sensor_names:
                print(f"      - {sensor[:60]:60s}: {sensor_counts[sensor]:,}")

    print(f"\n✅ Finished reading {total_rows:,} rows")

    # Process and save each sensor
    print(f"\n{'='*80}")
    print("💾 Processing and saving sensors")
    print(f"{'='*80}")

    results = {}

    for sensor in sensor_names:
        print(f"\n📦 {sensor}")

        if not sensor_data[sensor]:
            print(f"   ⚠️  No data found")
            continue

        # Combine chunks
        print(f"   🔗 Combining {len(sensor_data[sensor])} chunks...")
        df = pd.concat(sensor_data[sensor], ignore_index=True)

        # Parse timestamps
        print(f"   ⏰ Parsing timestamps...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

        # Clean
        before = len(df)
        df = df.dropna()
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna()
        after = len(df)

        print(f"   🧹 Cleaned: {before:,} → {after:,} rows")

        # Save
        safe_name = sensor.replace('/', '_').replace(' ', '_')
        output_csv = output_dir / f"{safe_name}.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

        # Stats
        duration = df['timestamp'].max() - df['timestamp'].min()
        print(f"   ✅ Saved to: {output_csv.name}")
        print(f"   📅 Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   ⏱️  Duration: {duration.days} days")
        print(f"   📊 Values: {df['value'].min():.2f} to {df['value'].max():.2f}")
        print(f"   📏 Mean: {df['value'].mean():.2f}, Std: {df['value'].std():.2f}")

        results[sensor] = {
            'output_file': output_csv,
            'measurements': len(df),
            'duration_days': duration.days
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract sensors from IoT dataset to ThesisML format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract single sensor
  python extract_sensors.py \
    --input new_iot_dataset/measurements-23-10-25.csv \
    --sensor "ZWave2_Metered_Wall_Plug_Switch_computers_Sensor_power" \
    --output extracted_sensors/

  # Extract multiple sensors from file
  python extract_sensors.py \
    --input new_iot_dataset/measurements-23-10-25.csv \
    --sensors-file selected_sensors.txt \
    --output extracted_sensors/
        """
    )

    parser.add_argument('--input', required=True,
                       help='Input CSV file (measurements)')
    parser.add_argument('--output', required=True,
                       help='Output directory for extracted sensors')
    parser.add_argument('--sensor', type=str,
                       help='Single sensor name to extract')
    parser.add_argument('--sensors', nargs='+',
                       help='Multiple sensor names to extract')
    parser.add_argument('--sensors-file', type=str,
                       help='File with sensor names (one per line)')
    parser.add_argument('--chunk-size', type=int, default=100000,
                       help='Chunk size for reading (default: 100000)')

    args = parser.parse_args()

    input_csv = Path(args.input)
    output_dir = Path(args.output)

    if not input_csv.exists():
        print(f"❌ Input file not found: {input_csv}")
        sys.exit(1)

    # Determine sensors to extract
    sensors = []

    if args.sensor:
        sensors.append(args.sensor)

    if args.sensors:
        sensors.extend(args.sensors)

    if args.sensors_file:
        sensors_file = Path(args.sensors_file)
        if sensors_file.exists():
            with open(sensors_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        sensors.append(line)

    if not sensors:
        print("❌ No sensors specified. Use --sensor, --sensors, or --sensors-file")
        sys.exit(1)

    # Remove duplicates
    sensors = list(dict.fromkeys(sensors))

    # Extract
    if len(sensors) == 1:
        output_csv = output_dir / f"{sensors[0].replace('/', '_').replace(' ', '_')}.csv"
        extract_sensor(input_csv, sensors[0], output_csv, args.chunk_size)
    else:
        extract_multiple_sensors(input_csv, sensors, output_dir, args.chunk_size)

    print(f"\n{'='*80}")
    print("✅ EXTRACTION COMPLETE!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
