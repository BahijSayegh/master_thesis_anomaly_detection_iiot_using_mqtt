import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import orjson
import zstandard as zstd
from scapy.all import load_contrib
from scapy.utils import PcapReader

from config.extractor_config import (RAW_DIR, PROCESSED_DIR, TARGET_CATEGORY, TARGET_PROCESS,
                                     SMALL_WINDOW, LARGE_WINDOW, COMPRESSION_LEVEL,
                                     FILES_TO_PROCESS, MAX_WORKERS)
from scripts.engineered_flow_features import process_windows, verify_feature_extraction
from scripts.label_processor import process_file_labels
from scripts.packet_level_features import process_packet
from scripts.protocol_aggregates import process_windows as process_protocol_features, verify_protocol_features
from scripts.sliding_window_processor import create_processor
from scripts.statistical_features import process_windows as process_statistical_features, verify_statistical_features
from scripts.temporal_features import process_windows as process_temporal_features, verify_temporal_features

# Load MQTT contribution
load_contrib('mqtt')

# Setup process pool
if MAX_WORKERS == -1:
    MAX_WORKERS = mp.cpu_count()


def setup_logging(process_dir: Path) -> Path:
    """Setup logging for a process directory."""
    log_file = process_dir / f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    return log_file


def get_output_path(input_file: Path, raw_dir: Path, processed_dir: Path) -> Path:
    """Generate output path maintaining directory structure."""
    relative_path = input_file.relative_to(raw_dir)
    output_path = processed_dir / relative_path
    output_path = output_path.with_suffix('.jsonl.zst')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def process_packet_with_windows(
        packet_dict: Dict[str, Any],
        small_window: List[Dict[str, Any]],
        large_window: List[Dict[str, Any]],
        packet_count: int
) -> Dict[str, Any]:
    """Process a packet with both window contexts to extract all features."""
    features = packet_dict.get('features', {})

    try:
        flow_features = process_windows(small_window, large_window)
        temporal_features = process_temporal_features(small_window, large_window)
        protocol_features = process_protocol_features(small_window, large_window)
        statistical_features = process_statistical_features(small_window, large_window)

        return {
            **features,
            **flow_features,
            **temporal_features,
            **protocol_features,
            **statistical_features,
            'packet_count': packet_count,
            'timestamp': packet_dict['time']
        }

    except Exception as e:
        print(f"Error in feature extraction pipeline: {e}")
        return {
            **features,
            'packet_count': packet_count,
            'timestamp': packet_dict['time']
        }


def process_single_file(args: Tuple[Path, Path, Path, int]) -> Dict[str, Any]:
    """
    Process a single pcapng file and extract features.
    
    Args:
        args: Tuple containing (input_file, output_file, log_file, process_id)
    
    Returns:
        Dict containing processing statistics
    """
    input_file, output_file, log_file, process_id = args

    stats = {
        'file_name': input_file.name,
        'packet_count': 0,
        'success': False,
        'error': None,
        'start_time': datetime.now(),
        'process_id': process_id
    }

    try:
        # Process labels
        labels = process_file_labels(str(input_file))
        if not labels:
            stats['error'] = "Failed to process labels"
            return stats

        # Initialize processor
        processor = create_processor(LARGE_WINDOW, SMALL_WINDOW)
        previous_time = None

        # Setup compression
        zctx = zstd.ZstdCompressor(level=COMPRESSION_LEVEL)

        # Process packets
        with PcapReader(str(input_file)) as pcap_reader, \
                open(output_file, 'wb') as raw_file, \
                zctx.stream_writer(raw_file) as comp_file:

            for packet in pcap_reader:
                try:
                    stats['packet_count'] += 1

                    features = process_packet(packet=packet, previous_time=previous_time)
                    current_time = float(packet.time)

                    packet_dict = {
                        'number': stats['packet_count'],
                        'time': current_time,
                        'length': len(packet),
                        'features': features
                    }

                    windows = processor.add_packet(packet_dict)
                    if windows:
                        small_window, large_window = windows
                        combined_features = process_packet_with_windows(
                            packet_dict=packet_dict,
                            small_window=small_window,
                            large_window=large_window,
                            packet_count=stats['packet_count']
                        )

                        if (verify_feature_extraction(combined_features) and
                                verify_temporal_features(combined_features) and
                                verify_protocol_features(combined_features) and
                                verify_statistical_features(combined_features)):
                            json_line = orjson.dumps({**labels, **combined_features}) + b'\n'
                            comp_file.write(json_line)

                    previous_time = current_time

                except Exception as e:
                    stats['error'] = f"Error processing packet {stats['packet_count']}: {str(e)}"
                    continue

        stats['success'] = True
        stats['end_time'] = datetime.now()

    except Exception as e:
        stats['error'] = str(e)
        stats['end_time'] = datetime.now()

    return stats


def log_result(result: Dict[str, Any], log_file: Path) -> None:
    """Log the result of file processing."""
    elapsed_time = (result['end_time'] - result['start_time']).total_seconds()
    rate = result['packet_count'] / elapsed_time if elapsed_time > 0 else 0

    log_message = (
        f"Process {result['process_id']}: {result['file_name']}\n"
        f"{'Success' if result['success'] else 'Failed'}: "
        f"{result['packet_count']} packets processed in {elapsed_time:.1f} seconds "
        f"({rate:.1f} packets/sec)\n"
    )

    if result['error']:
        log_message += f"Error: {result['error']}\n"

    log_message += "-" * 80 + "\n"

    with open(log_file, 'a') as f:
        f.write(log_message)
    print(log_message, end='')


def process_directory(category: str, process_type: str) -> None:
    """
    Process all pcapng files in a directory using parallel processing.

    Args:
        category: "Normal" or "Anomalous"
        process_type: "Order" or "Stock"
    """
    # Setup paths
    input_dir = RAW_DIR / category / process_type
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        return

    # Setup logging
    log_file = setup_logging(PROCESSED_DIR / category / process_type)

    # Get list of all pcapng files
    pcap_files = list(input_dir.glob('*.pcapng'))
    total_files = len(pcap_files)

    # Determine number of files to process
    files_to_process = total_files if FILES_TO_PROCESS == -1 else min(FILES_TO_PROCESS, total_files)
    files_to_process_list = pcap_files[:files_to_process]

    # Log processing start
    start_message = (
        f"\nProcessing directory: {input_dir}\n"
        f"Total files available: {total_files}\n"
        f"Files to process: {files_to_process}\n"
        f"Using {MAX_WORKERS} parallel processes\n\n"
    )
    print(start_message, end='')
    with open(log_file, 'a') as f:
        f.write(start_message)

    # Prepare arguments for parallel processing
    process_args = [
        (
            input_file,
            get_output_path(input_file, RAW_DIR, PROCESSED_DIR),
            log_file,
            i
        )
        for i, input_file in enumerate(files_to_process_list)
    ]

    # Process files in parallel
    with mp.Pool(MAX_WORKERS) as pool:
        for result in pool.imap_unordered(process_single_file, process_args):
            log_result(result, log_file)

    # Log completion
    end_message = (
        f"\nDirectory processing complete: {category}/{process_type}\n"
        f"Processed {files_to_process} out of {total_files} files\n\n"
    )
    print(end_message, end='')
    with open(log_file, 'a') as f:
        f.write(end_message)


def main():
    """Main processing function."""
    if TARGET_PROCESS == "ALL":
        for process_type in ["Order", "Stock"]:
            process_directory(TARGET_CATEGORY, process_type)
    else:
        process_directory(TARGET_CATEGORY, TARGET_PROCESS)


if __name__ == "__main__":
    main()
