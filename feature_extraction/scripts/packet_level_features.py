from dataclasses import dataclass
from typing import Dict, Any, Optional
from scapy.contrib.mqtt import *
from scapy.layers.inet import IP
from scapy.layers.l2 import Ether

# Bind MQTT to TCP ports
bind_layers(TCP, MQTT, sport=1883)
bind_layers(TCP, MQTT, dport=1883)
bind_layers(TCP, MQTT, sport=8883)
bind_layers(TCP, MQTT, dport=8883)

# Constants
MQTT_PORTS = {1883, 8883}


@dataclass
class PacketFeatures:
    """Dataclass for storing packet features with default values."""

    # TCP/IP Features
    packet_size: int = 0
    inter_arrival_time: float = 0.0
    source_ip_address: str = ""
    destination_ip_address: str = ""
    source_port: int = 0
    destination_port: int = 0
    tcp_syn_flag: int = 0
    tcp_ack_flag: int = 0
    tcp_fin_flag: int = 0
    tcp_rst_flag: int = 0
    tcp_psh_flag: int = 0
    tcp_urg_flag: int = 0
    tcp_ece_flag: int = 0
    tcp_cwr_flag: int = 0
    tcp_payload_size: int = 0
    tcp_window_size: int = 0
    tcp_sequence_number: int = 0
    tcp_acknowledgment_number: int = 0
    source_mac_address: str = ""
    destination_mac_address: str = ""

    # MQTT Features
    is_mqtt: int = 0
    mqtt_message_type: str = ""
    mqtt_qos_level: int = 0
    mqtt_dup_flag: int = 0
    mqtt_retain_flag: int = 0
    mqtt_clean_session_flag: int = 0
    mqtt_will_flag: int = 0
    mqtt_will_qos: int = 0
    mqtt_will_retain: int = 0
    mqtt_topic: str = ""
    mqtt_payload_size: int = 0
    mqtt_message_timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary format."""
        return {k: v for k, v in self.__dict__.items()}


def extract_tcp_ip_features(packet: Packet) -> Dict[str, Any]:
    """
    Extract TCP/IP features from a Scapy packet.
    """

    features = PacketFeatures()
    try:
        # Basic packet features
        features.packet_size = len(packet)

        # IP Layer features
        if IP in packet:
            ip = packet[IP]
            features.source_ip_address = ip.src
            features.destination_ip_address = ip.dst

        # TCP Layer features
        if TCP in packet:
            tcp = packet[TCP]
            features.source_port = tcp.sport
            features.destination_port = tcp.dport
            features.tcp_window_size = tcp.window
            features.tcp_sequence_number = tcp.seq
            features.tcp_acknowledgment_number = tcp.ack
            features.tcp_payload_size = len(tcp.payload) if tcp.payload else 0

            # TCP Flags
            features.tcp_syn_flag = int(tcp.flags.S)
            features.tcp_ack_flag = int(tcp.flags.A)
            features.tcp_fin_flag = int(tcp.flags.F)
            features.tcp_rst_flag = int(tcp.flags.R)
            features.tcp_psh_flag = int(tcp.flags.P)
            features.tcp_urg_flag = int(tcp.flags.U)
            features.tcp_ece_flag = int(tcp.flags.E)
            features.tcp_cwr_flag = int(tcp.flags.C)

        # Ethernet Layer features
        if Ether in packet:
            eth = packet[Ether]
            features.source_mac_address = eth.src
            features.destination_mac_address = eth.dst

    except Exception as e:
        print(f"Error extracting TCP/IP features: {e}")

    return features.to_dict()


def is_mqtt_packet(packet: Packet) -> bool:
    """
    Check if packet is an MQTT packet.
    """

    if TCP not in packet:
        return False

    # Check MQTT ports
    tcp = packet[TCP]
    if tcp.sport not in MQTT_PORTS and tcp.dport not in MQTT_PORTS:
        return False

    # Try to decode as MQTT
    try:
        if tcp.payload:
            decoded = MQTT(bytes(tcp.payload))
            return True if decoded.type is not None else False

    except Exception:
        pass

    return False


def extract_mqtt_features(packet: Packet, tcp_ip_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract MQTT protocol features using Scapy's MQTT contribution module.

    Args:
        packet: Scapy packet
        tcp_ip_features: Dictionary of TCP/IP features

    Returns:
        Dict[str, Any]: Dictionary of MQTT features

    Raises:
        RuntimeError: If MQTT packet parsing fails
        ValueError: If field extraction fails
        UnicodeDecodeError: If topic decoding fails
    """
    features = PacketFeatures(**tcp_ip_features)

    if not is_mqtt_packet(packet):
        return features.to_dict()

    try:
        # Force MQTT decoding
        tcp_payload = bytes(packet[TCP].payload)
        mqtt_packet = MQTT(tcp_payload)
    except (AttributeError, TypeError) as e:
        print(f"Failed to decode MQTT packet: {str(e)}")
        return features.to_dict()

    try:
        # Mark as MQTT packet and set timestamp
        features.is_mqtt = 1
        features.mqtt_message_timestamp = float(packet.time)

        # Extract type-specific features
        msg_type = mqtt_packet.type
        features.mqtt_message_type = str(msg_type)

        # Process different message types
        if msg_type == 1:  # CONNECT
            _process_connect(mqtt_packet, features)
        elif msg_type == 3:  # PUBLISH
            _process_publish(mqtt_packet, features)
        elif msg_type == 8:  # SUBSCRIBE
            _process_subscribe(mqtt_packet, features)
        elif msg_type == 10:  # UNSUBSCRIBE
            _process_unsubscribe(mqtt_packet, features)

    except AttributeError as e:
        print(f"Missing MQTT attribute in packet type {mqtt_packet.type}: {str(e)}")
        features.is_mqtt = 0
    except UnicodeDecodeError as e:
        print(f"Failed to decode MQTT topic in packet type {mqtt_packet.type}: {str(e)}")
    except ValueError as e:
        print(f"Invalid value in MQTT packet type {mqtt_packet.type}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error processing MQTT packet type {mqtt_packet.type}: {str(e)}")
        features.is_mqtt = 0

    return features.to_dict()


def _process_connect(mqtt_packet: Packet, features: PacketFeatures) -> None:
    """Process CONNECT packet type."""
    try:
        connect = mqtt_packet.getlayer(MQTTConnect)
        if not connect:
            return

        # Direct attribute access with default values
        features.mqtt_clean_session_flag = int(connect.clean_session)
        features.mqtt_will_flag = int(connect.will)
        features.mqtt_will_qos = int(connect.willqos)
        features.mqtt_will_retain = int(connect.willretain)

    except AttributeError:
        features.mqtt_clean_session_flag = features.mqtt_will_flag = features.mqtt_will_qos = features.mqtt_will_retain = 0


def _process_publish(mqtt_packet: Packet, features: PacketFeatures) -> None:
    """Process PUBLISH packet type."""
    try:
        publish = mqtt_packet.getlayer(MQTTPublish)
        if not publish:
            return

        # Direct attribute access
        features.mqtt_qos_level = int(publish.qos)
        features.mqtt_dup_flag = int(publish.dup)
        features.mqtt_retain_flag = int(publish.retain)
        features.mqtt_topic = publish.topic.decode('utf-8', errors='ignore') if isinstance(publish.topic, bytes) else str(publish.topic)
        features.mqtt_payload_size = len(publish.value) if publish.value else 0

    except AttributeError:
        features.mqtt_qos_level = features.mqtt_dup_flag = features.mqtt_retain_flag = features.mqtt_payload_size = 0
        features.mqtt_topic = ""


def _process_subscribe(mqtt_packet: Packet, features: PacketFeatures) -> None:
    """Process SUBSCRIBE packet type."""
    try:
        subscribe = mqtt_packet.getlayer(MQTTSubscribe)
        if not subscribe or not subscribe.topics:
            features.mqtt_topic = ""
            features.mqtt_qos_level = 0
            return

        # Direct access to first topic
        topic = subscribe.topics[0]
        features.mqtt_topic = topic.topic.decode('utf-8', errors='ignore') if isinstance(topic.topic, bytes) else str(topic.topic)
        features.mqtt_qos_level = int(topic.qos)

    except (AttributeError, IndexError):
        features.mqtt_topic = ""
        features.mqtt_qos_level = 0


def _process_unsubscribe(mqtt_packet: Packet, features: PacketFeatures) -> None:
    """Process UNSUBSCRIBE packet type."""
    try:
        unsubscribe = mqtt_packet.getlayer(MQTTUnsubscribe)
        if not unsubscribe or not unsubscribe.topics:
            features.mqtt_topic = ""
            return

        # Direct access to first topic
        topic = unsubscribe.topics[0]
        features.mqtt_topic = topic.decode('utf-8', errors='ignore') if isinstance(topic, bytes) else str(topic)

    except (AttributeError, IndexError):
        features.mqtt_topic = ""


def process_packet(
        packet: Packet,
        previous_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Process a packet to extract all features.
    """

    try:
        # Extract TCP/IP features
        features = extract_tcp_ip_features(packet)

        # Extract MQTT features if present
        if TCP in packet and (features['source_port'] in MQTT_PORTS or
                              features['destination_port'] in MQTT_PORTS):
            features = extract_mqtt_features(packet, features)

        # Calculate inter-arrival time
        if previous_time is not None:
            features['inter_arrival_time'] = float(packet.time) - previous_time

        return features

    except Exception as e:
        print(f"Error in main feature extraction: {e}")
        return PacketFeatures().to_dict()
