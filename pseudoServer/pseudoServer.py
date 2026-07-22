# To do in this pseudoServer:

# Heartbeat controller (so a protocol for that)
# Ring formation and management 
# Start the training Process

# Once this is possible then look to add the pytorch file transfer 
# and also the datasetshard transfer

#heartbeat controller is basically send 1 byte of info so a special symbol means
#its a hello packet

import time
import struct
import socket
import threading
import os
from dataclasses import dataclass, field

BUFFER = 4096
HOST = "127.0.0.1"
PORT = 9999

PEER_INFO = b'\x00'
PEER_INQUIRY = b'\xff'
PEER_ALIVE = b'\xbb'
START = b'\xcc'

@dataclass
class Worker:
    ip_addr : str
    listening_port : int
    fault_port : int
    # hello_port : int
    last_update_time : float = field(default_factory=time.time)
        
peer_list = []
peer_list_lock = threading.Lock()

def peer_alive_check():
    curr_time = time.time()
    survivors = []
    for peer in peer_list:
        if curr_time - peer.last_update_time < 30.0:
            survivors.append(peer)

    if len(peer_list) != len(survivors):
        peer_list[:] = survivors
        ring_formation(peer_list)

def peer_alive_check_thread_fn():
    while True:
        peer_alive_check()
        time.sleep(10)

#when start is initiated then this func is called
def ring_formation(peer_list):
    peer_list_len = len(peer_list)
    for i in range(peer_list_len):
        current_peer = peer_list[i]
        sending_peer = peer_list[(i + 1) % peer_list_len]
        receiving_peer = peer_list[(i - 1 + peer_list_len) % peer_list_len]

        sending_ip_bytes = socket.inet_aton(sending_peer.ip_addr)
        receiving_ip_bytes = socket.inet_aton(receiving_peer.ip_addr)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ring_socket:
            ring_socket.connect((current_peer.ip_addr, current_peer.fault_port))

            message = struct.pack(
                '!c4sH4sH',
                START,
                sending_ip_bytes, 
                sending_peer.listening_port, 
                receiving_ip_bytes, 
                receiving_peer.listening_port
            )
            
            ring_socket.sendall(message)

def exit_thread_fn():
    while True:
        command = input("$ ")
        if "exit" in command:
            os._exit(0)

def recieve_data(peer_conn, message_size):
    message = b""
    temp_msg_size = message_size
    recv_size = BUFFER
    while True:
        if temp_msg_size <= min(temp_msg_size, BUFFER):
            message_frag = peer_conn.recv(temp_msg_size)
            recv_size = len(message_frag)
            message += message_frag

            if recv_size < temp_msg_size:
                temp_msg_size -= recv_size
                continue
            break
            
        message_frag = peer_conn.recv(BUFFER)
        recv_size = len(message_frag)
        message += message_frag
        temp_msg_size -= recv_size

    return message

def find_peer_by_ip(peer_addr):
    for peer in peer_list:
        if peer.ip_addr == peer_addr[0]:
            return peer
    return None

def peer_thread_function(peer_conn, peer_addr):
    with peer_conn:
        packet_status = peer_conn.recv(1)
        
        # message_str = recieve_data(peer_conn, message_size)

        if packet_status == PEER_ALIVE:
            new_time = time.time()
            
            curr_peer = find_peer_by_ip(peer_addr)

            if curr_peer != None:
                curr_peer.last_update_time = new_time

        if packet_status == PEER_INFO:
            message_size, = struct.unpack('!I', peer_conn.recv(4))
            listening_port, fault_port = struct.unpack('!HH', peer_conn.recv(message_size))

            peer_info = Worker(
                ip_addr=peer_addr[0], 
                listening_port=listening_port, 
                fault_port = fault_port,
                last_update_time= time.time()
            )
            
            with peer_list_lock:
                peer_list.append(peer_info)
            # print(peer_list)

        # if packet_status == PEER_INQUIRY:
        #     #iterate thorough peer list
        #     for i in range(len(peer_list)):
        #         peer_data = peer_list[i].split("|")
        #         file_list = peer_data[2].split(" ")

        #         results = [(peer_data[0], peer_data[1]) for item in file_list if item == message_str]

        #         if results:
        #             dpeer_ip, dpeer_port = results[0]
            
        #     dpeer_details = f"{dpeer_ip}|{dpeer_port}"
        #     dpeer_details_b = dpeer_details.encode('utf-8')

        #     header = PEER_INFO + len(dpeer_details_b).to_bytes(4, byteorder='big')
        #     peer_conn.sendall(header + dpeer_details_b)

def start_func():
    while True:
        command = input("$ ")
        if "start" in command:
            ring_formation(peer_list)

# exit_thread = threading.Thread(target=exit_thread_fn)
# exit_thread.start()

start_thread = threading.Thread(target=start_func)
start_thread.start()

peer_alive_check_thread = threading.Thread(target=peer_alive_check_thread_fn)
peer_alive_check_thread.start()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    while True:
        peer_conn, peer_addr = server_socket.accept()

        peer_thread = threading.Thread(target=peer_thread_function, args=(peer_conn, peer_addr))
        peer_thread.start()
    