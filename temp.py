from scapy.all import PcapReader, Dot11

file_path = "train_pcap/camera_youtube.pcap"

stream_dict = {}
for pkt in PcapReader(file_path):
    if pkt.haslayer(Dot11):
        dot11 = pkt[Dot11]
        if dot11.type == 2:
            src_mac = dot11.addr2
            dst_mac = dot11.addr1
            key = (src_mac, dst_mac)
            stream_dict.setdefault(key, 0)
            stream_dict[key] += 1

print("전체 스트림 수:", len(stream_dict))
print(
    "상위 10개 스트림 (패킷수):", sorted(stream_dict.items(), key=lambda x: -x[1])[:10]
)
