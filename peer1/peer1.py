import socket
import threading
import os
import subprocess
import struct
import time

BUFFER = 4096

SERVER_IP = "127.0.0.1"
SERVER_PORT = 9999

WORKER_IP = "127.0.0.1"
LISTENING_PORT = 37000
DATASET_PORT = 12000
FAULT_PORT = 20000

PEER_INFO = b'\x00'
PEER_INQUIRY = b'\xff'
DATASET_DOWNLOAD = b'\xaa'
DATASET_CONTENT = b'\x77'
START = b'\xcc'
PEER_ALIVE = b'\xbb'

#initially the addresses are empty
send_to_addr = (0, 0)
recv_from_addr = (0, 0)

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

def start_listen():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as peer_socket:
        peer_socket.bind((WORKER_IP, FAULT_PORT))
        peer_socket.listen()

        while True:    
            #node refers to other peers that are trying to connect with this peer but here server too can connect for initial starting
            node_conn, node_addr = peer_socket.accept()

            if node_addr[0] != SERVER_IP:
                node_conn.close()
                continue

            #get the file name and then send the file
            packet_status = node_conn.recv(1)

            if packet_status == START:
                sending_ip_bytes, send_to_port, receiving_ip_bytes, recv_from_port = struct.unpack('!4sH4sH', node_conn.recv(12))

                send_to_ip = socket.inet_ntoa(sending_ip_bytes)
                recv_from_ip = socket.inet_ntoa(receiving_ip_bytes)

                send_to_addr = (send_to_ip, send_to_port)
                recv_from_addr = (recv_from_ip, recv_from_port)

                print(send_to_addr)
                print(recv_from_addr)

            node_conn.close()

#for gradient transfer
def listen_from_worker():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as worker_socket:
        worker_socket.bind(WORKER_IP, LISTENING_PORT)
        worker_socket.listen()

        node_conn, node_addr = worker_socket.accept()

        if node_addr[0] != recv_from_addr[0]:
            node_conn.close()

        #if its the connection that is actually needed then we do the file transfer stuffs
        message = node_conn.recv(7)
        print(message.decode('utf-8'))

def send_to_worker():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as worker_socket:
        worker_socket.connect(send_to_addr)

        # send the the stuffs needed to be sent
        worker_socket.sendall(b"GRADIENT")

# Hello packet to indicate that the worker is still alive
def hello_packet_fn():
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as peer_socket:
            peer_socket.bind((WORKER_IP, 0))
            peer_socket.connect((SERVER_IP, SERVER_PORT))
            peer_socket.sendall(PEER_ALIVE)
        
        time.sleep(10)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as peer_socket:
    peer_socket.bind((WORKER_IP, 0))
    peer_socket.connect((SERVER_IP, SERVER_PORT))

    # sending peer_info packet
    message = struct.pack("!HH", LISTENING_PORT, FAULT_PORT)
    message_size = len(message)
    
    header = struct.pack("!cI", PEER_INFO, message_size)
    peer_socket.sendall(header + message)


#after initial setup the peer should listen for reqs as well so createing a thread for that
start_listening_thread = threading.Thread(target=start_listen)
start_listening_thread.start()

#hello packet sending thread
hello_thread = threading.Thread(target=hello_packet_fn)
hello_thread.start()