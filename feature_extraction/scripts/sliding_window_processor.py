import traceback
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from config.extractor_config import SMALL_WINDOW, LARGE_WINDOW


class SlidingWindowProcessor:
    """
    Sliding window processor for network packet analysis.

    Attributes:
        buffer: Circular buffer for packet storage
        large_window_size: Size of large window (default 500)
        small_window_size: Size of small window (default 100)
        packets_processed: Counter for processed packets
        windows_generated: Counter for generated windows
    """

    def __init__(self, large_window_size: int = LARGE_WINDOW,
                 small_window_size: int = SMALL_WINDOW):
        """
        Initialize the sliding window processor.

        Args:
            large_window_size: Size of larger window (default from config)
            small_window_size: Size of smaller window (default from config)

        Raises:
            ValueError: If window sizes are invalid
        """

        if small_window_size > large_window_size:
            raise ValueError("Small window size cannot be larger than large window size")

        # Initialize buffers and counters
        self.buffer = deque(maxlen=large_window_size)
        self.packets_processed = 0
        self.windows_generated = 0

        # Window configuration
        self.large_window_size = large_window_size
        self.small_window_size = small_window_size

        # Pre-allocate numpy arrays for feature calculation
        self.packet_sizes = np.zeros(large_window_size, dtype=np.float64)
        self.timestamps = np.zeros(large_window_size, dtype=np.float64)

    def buffer_ready(self) -> bool:
        """
        Check if buffer has enough packets for window generation.

        Returns:
            bool: True if buffer is ready for window generation
        """

        return len(self.buffer) == self.large_window_size

    def add_packet(self, packet: Dict[str, Any]) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """
        Add a new packet and generate windows if buffer is ready.

        Args:
            packet: Packet dictionary containing at minimum:
                - 'number': Packet sequence number
                - 'time': Packet timestamp
                - 'length': Packet length
                - 'features': Dictionary of packet features

        Returns:
            Optional[Tuple[List[Dict], List[Dict]]]:
                If buffer ready: (small_window, large_window)
                If still buffering: None
        """

        try:
            # Update buffers
            self.buffer.append(packet)
            buffer_idx = self.packets_processed % self.large_window_size
            self.packet_sizes[buffer_idx] = float(packet['length'])
            self.timestamps[buffer_idx] = float(packet['time'])
            self.packets_processed += 1

            # Generate windows if buffer is ready
            if self.buffer_ready():
                self.windows_generated += 1
                return self._get_windows()

            return None

        except Exception as e:
            print(f"Error in add_packet: {e}")
            traceback.print_exc()

            return None

    def _get_windows(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate both windows efficiently.

        Returns:
            Tuple[List[Dict], List[Dict]]: (small_window, large_window)
        """

        try:
            buffer_list = list(self.buffer)
            small_window = buffer_list[-self.small_window_size:]
            return small_window, buffer_list

        except Exception as e:
            print(f"Error in _get_windows: {e}")
            traceback.print_exc()
            raise

    def get_window_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get numpy arrays of packet sizes and timestamps.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (packet_sizes, timestamps)
        """

        return (
            self.packet_sizes.copy(),
            self.timestamps.copy()
        )

    def get_target_packet(self) -> Optional[Dict[str, Any]]:
        """
        Get the current target packet (most recent).

        Returns:
            Optional[Dict[str, Any]]: Most recent packet or None if buffer empty
        """

        return self.buffer[-1] if self.buffer else None

    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get current processing status and statistics.

        Returns:
            Dict[str, Any]: Status dictionary containing:
                - packets_processed: Total packets processed
                - windows_generated: Total windows generated
                - buffer_size: Current buffer size
                - buffer_complete: Whether buffer is full
                - window_sizes: Configured window sizes
        """

        return {
            'packets_processed': self.packets_processed,
            'windows_generated': self.windows_generated,
            'buffer_size': len(self.buffer),
            'buffer_complete': self.buffer_ready(),
            'large_window_size': self.large_window_size,
            'small_window_size': self.small_window_size
        }

    def clear(self) -> None:
        """
        Clear all buffers and reset counters.
        """

        self.buffer.clear()
        self.packet_sizes.fill(0)
        self.timestamps.fill(0)


def create_processor(large_window_size: int = LARGE_WINDOW,
                     small_window_size: int = SMALL_WINDOW) -> SlidingWindowProcessor:
    """
    Factory function to create a sliding window processor.

    Args:
        large_window_size: Size of larger window
        small_window_size: Size of smaller window

    Returns:
        SlidingWindowProcessor: Configured processor instance

    Raises:
        ValueError: If window sizes are invalid
    """

    return SlidingWindowProcessor(
        large_window_size=large_window_size,
        small_window_size=small_window_size
    )
