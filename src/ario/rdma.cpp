#include "ario/rdma.h"

#include "ario/utils.h"

namespace ario {

constexpr size_t kRdmaBufferPoolBits = 30;
constexpr size_t kRdmaBufferBlockBits = 20;

RdmaConnector::RdmaConnector(std::string dev_name,
                             std::shared_ptr<EventHandler> handler)
    : dev_name_(std::move(dev_name)),
      handler_(std::move(handler)),
      tcp_acceptor_(executor_) {
  // TODO: RAII
  int ret;
  CreateContext();
}

RdmaConnector::~RdmaConnector() {
  connections_.clear();
  if (ctx_) ibv_close_device(ctx_);
}

void RdmaConnector::CreateContext() {
  int cnt, ret;
  ibv_context *ctx = nullptr;
  ibv_device **devs = ibv_get_device_list(&cnt);
  for (int i = 0; i < cnt; i++) {
    const char *name = ibv_get_device_name(devs[i]);
    ctx = ibv_open_device(devs[i]);
    ibv_device_attr device_attr;
    ret = ibv_query_device(ctx, &device_attr);
    if (ret) die("ibv_query_device");
    fprintf(stderr,
            "Found ibv device: name=%s, guid=0x%016lx. Active ports:", name,
            ibv_get_device_guid(devs[i]));
    int dev_port = 0;
    for (int p = 1; p <= device_attr.phys_port_cnt; ++p) {
      ibv_port_attr port_attr;
      ret = ibv_query_port(ctx, p, &port_attr);
      if (ret) die("ibv_query_port");
      if (port_attr.state != IBV_PORT_ACTIVE) {
        continue;
      }
      fprintf(stderr, " %d", p);
      if (!dev_port) {
        dev_port = p;
      }
    }
    fprintf(stderr, "\n");

    if (dev_name_ == name) {
      if (!dev_port) die("Could not find active port at device " + dev_name_);
      ctx_ = ctx;
      dev_port_ = dev_port;
    } else {
      ibv_close_device(ctx);
    }
  }
  ibv_free_device_list(devs);
  if (!ctx_) die("Could not open device: " + dev_name_);
  fprintf(stderr, "Opened ibv device %s at port %d\n", dev_name_.c_str(),
          dev_port_);
}

void RdmaConnector::ListenTcp(uint16_t port,
                              std::vector<uint8_t> &memory_region) {
  memory_region_ = &memory_region;
  tcp_acceptor_.BindAndListen(port);
  fprintf(stderr, "TCP server listening on port %d\n", port);
  TcpAccept();
}

void RdmaConnector::TcpAccept() {
  tcp_acceptor_.AsyncAccept([this](int error, TcpSocket peer) {
    if (error) {
      fprintf(stderr, "TcpAccept error=%d\n", error);
      die("TcpAccept AsyncAccept");
    }

    AddConnection(std::move(peer));
    TcpAccept();
  });
}

void RdmaConnector::ConnectTcp(const std::string &host, uint16_t port) {
  fprintf(stderr, "Connecting TCP to host %s port %u\n", host.c_str(), port);
  tcp_socket_.Connect(executor_, host, port);
  fprintf(stderr, "TCP socket connected\n");
  AddConnection(std::move(tcp_socket_));
}

void RdmaConnector::RunEventLoop() { executor_.RunEventLoop(); }

void RdmaConnector::StopEventLoop() { executor_.StopEventLoop(); }

Connection *RdmaConnector::GetConnection() {
  if (connections_.empty()) return nullptr;
  return connections_.front().get();
}

void RdmaConnector::AddConnection(TcpSocket tcp) {
  auto conn = new Connection(dev_name_, dev_port_, std::move(tcp), ctx_,
                             memory_region_, handler_);
  connections_.emplace_back(conn);
}

Connection::Connection(std::string dev_name, int dev_port, TcpSocket tcp,
                       ibv_context *ctx, std::vector<uint8_t> *memory_region,
                       std::shared_ptr<EventHandler> handler)
    : dev_name_(std::move(dev_name)),
      dev_port_(dev_port),
      handler_(std::move(handler)),
      memory_region_(memory_region),
      local_buf_(kRdmaBufferPoolBits, kRdmaBufferBlockBits),
      tcp_(std::move(tcp)),
      ctx_(ctx) {
  int ret;
  // poller_type_ = PollerType::kSpinning;
  poller_type_ = PollerType::kBlocking;
  is_connected_ = false;

  BuildProtectionDomain();
  BuildCompletionQueue();
  BuildQueuePair();
  TransitQueuePairToInit();
  RegisterMemory();
  SendConnInfo();
  RecvConnInfo();
}

Connection::~Connection() {
  poller_stop_ = true;
  cq_poller_thread_.join();
}

void Connection::BuildProtectionDomain() {
  pd_ = ibv_alloc_pd(ctx_);
  if (!pd_) die_perror("ibv_alloc_pd");
}

void Connection::BuildCompletionQueue() {
  constexpr int kNumCompletionQueueEntries = 100;
  int ret;

  if (poller_type_ == PollerType::kBlocking) {
    comp_channel_ = ibv_create_comp_channel(ctx_);
    if (!comp_channel_) die_perror("ibv_create_comp_channel");
    SetNonBlocking(comp_channel_->fd);
    comp_channel_pollfd_.fd = comp_channel_->fd;
    comp_channel_pollfd_.events = POLLIN;
    comp_channel_pollfd_.revents = 0;
  } else {
    comp_channel_ = nullptr;
  }

  cq_ = ibv_create_cq(ctx_, kNumCompletionQueueEntries, nullptr, comp_channel_,
                      0);
  if (!cq_) die_perror("ibv_create_cq");
}

void Connection::BuildQueuePair() {
  constexpr uint32_t kMaxSendQueueSize = 1024;
  constexpr uint32_t kMaxRecvQueueSize = 1024;
  constexpr uint32_t kMaxSendScatterGatherElements = 16;
  constexpr uint32_t kMaxRecvScatterGatherElements = 16;

  ibv_qp_init_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.send_cq = cq_;
  attr.recv_cq = cq_;
  attr.qp_type = IBV_QPT_RC;
  attr.cap.max_send_wr = kMaxSendQueueSize;
  attr.cap.max_recv_wr = kMaxRecvQueueSize;
  attr.cap.max_send_sge = kMaxSendScatterGatherElements;
  attr.cap.max_recv_sge = kMaxRecvScatterGatherElements;

  qp_ = ibv_create_qp(pd_, &attr);
  if (!qp_) die_perror("ibv_create_qp");
}

void Connection::SendConnInfo() {
  ibv_port_attr attr;
  int ret = ibv_query_port(ctx_, dev_port_, &attr);
  if (ret) die_perror("SendConnInfo: ibv_query_port");
  ibv_gid gid;
  memset(&gid, 0, sizeof(gid));
  if (!attr.lid) {
    // Only InfiniBand has Local ID. RoCE needs Global ID.
    ibv_query_gid(ctx_, dev_port_, 0, &gid);
    if (ret) die_perror("SendConnInfo: ibv_query_gid");
  }

  auto msg = std::make_shared<RdmaConnectorMessage>();
  msg->type = RdmaConnectorMessage::Type::kConnInfo;
  msg->payload.conn.lid = attr.lid;
  msg->payload.conn.gid = gid;
  msg->payload.conn.qp_num = qp_->qp_num;
  fprintf(stderr, "local ConnInfo: qp_num=%d, lid=%d, gid[0]=%016lx:%016lx\n",
          qp_->qp_num, attr.lid, gid.global.subnet_prefix,
          gid.global.interface_id);

  fprintf(stderr, "Sending ConnInfo\n");
  ConstBuffer buf(msg.get(), sizeof(*msg));
  tcp_.AsyncWrite(buf, [msg = std::move(msg)](int err, size_t) {
    if (err) {
      fprintf(stderr, "SendConnInfo: AsyncWrite err = %d\n", err);
      die("SendConnInfo AsyncWrite callback");
    }
    fprintf(stderr, "ConnInfo sent\n");
  });
}

void Connection::RecvConnInfo() {
  fprintf(stderr, "Waiting for peer ConnInfo\n");
  auto msg = std::make_shared<RdmaConnectorMessage>();
  MutableBuffer buf(msg.get(), sizeof(*msg));
  tcp_.AsyncRead(buf, [msg = std::move(msg), this](int err, size_t) {
    if (err) {
      fprintf(stderr, "RecvConnInfo: AsyncRead err = %d\n", err);
      die("RecvConnInfo AsyncRead callback");
    }
    if (msg->type != RdmaConnectorMessage::Type::kConnInfo) {
      fprintf(stderr, "RecvConnInfo: AsyncRead msg->type = %d\n",
              static_cast<int>(msg->type));
      die("RecvConnInfo AsyncRead callback");
    }
    fprintf(stderr, "Received peer ConnInfo\n");

    TransitQueuePairToRTR(msg->payload.conn);
    TransitQueuePairToRTS();
    if (memory_region_) {
      // Server
      MarkConnected();
      handler_->OnConnected(this);
      SendMemoryRegion();
    } else {
      // Client
      RecvMemoryRegion();
    }
  });
}

void Connection::SendMemoryRegion() {
  fprintf(stderr, "Sending MemoryRegion\n");
  auto msg = std::make_shared<RdmaConnectorMessage>();
  msg->type = RdmaConnectorMessage::Type::kMemoryRegion;
  msg->payload.mr.addr = reinterpret_cast<uint64_t>(rdma_remote_mr_->addr);
  msg->payload.mr.size = rdma_remote_mr_->length;
  msg->payload.mr.rkey = rdma_remote_mr_->rkey;

  ConstBuffer buf(msg.get(), sizeof(*msg));
  tcp_.AsyncWrite(buf, [msg = std::move(msg)](int err, size_t) {
    if (err) {
      fprintf(stderr, "SendMemoryRegion: AsyncWrite err = %d\n", err);
      die("SendMemoryRegion AsyncWrite callback");
    }
    fprintf(stderr, "MemoryRegion sent\n");
  });
}

void Connection::RecvMemoryRegion() {
  fprintf(stderr, "Waiting for peer MemoryRegion\n");
  auto msg = std::make_shared<RdmaConnectorMessage>();
  MutableBuffer buf(msg.get(), sizeof(*msg));
  tcp_.AsyncRead(buf, [msg = std::move(msg), this](int err, size_t) {
    if (err) {
      fprintf(stderr, "RecvMemoryRegion: AsyncRead err = %d\n", err);
      return;
    }
    if (msg->type != RdmaConnectorMessage::Type::kMemoryRegion) {
      fprintf(stderr, "RecvMemoryRegion: AsyncRead msg->type = %d\n",
              static_cast<int>(msg->type));
      return;
    }

    remote_mr_ = msg->payload.mr;
    fprintf(stderr, "got memory region: addr=0x%016lx, size=%lu, lkey=0x%08x\n",
            remote_mr_.addr, remote_mr_.size, remote_mr_.rkey);
    MarkConnected();
    handler_->OnConnected(this);
  });
}

void Connection::TransitQueuePairToInit() {
  ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = dev_port_;
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

  int ret = ibv_modify_qp(
      qp_, &attr,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  if (ret) die_perror("TransitQueuePairToInit");
}

void Connection::TransitQueuePairToRTR(
    const RdmaConnectorMessage::ConnInfo &msg) {
  fprintf(stderr, "remote ConnInfo: qp_num=%d, lid=%d, gid[0]=%016lx:%016lx\n",
          msg.qp_num, msg.lid, msg.gid.global.subnet_prefix,
          msg.gid.global.interface_id);
  ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.ah_attr.port_num = dev_port_;
  attr.ah_attr.dlid = msg.lid;
  attr.path_mtu = IBV_MTU_1024;
  attr.dest_qp_num = msg.qp_num;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;  // 0.640 ms

  if (msg.lid) {
    // Infiniband
    attr.ah_attr.dlid = msg.lid;
  } else {
    // RoCE
    attr.ah_attr.is_global = true;
    attr.ah_attr.grh.dgid = msg.gid;
    attr.ah_attr.grh.hop_limit = 1;
  }

  int ret = ibv_modify_qp(qp_, &attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                              IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (ret) die_perror("TransitQueuePairToRTR");
}

void Connection::TransitQueuePairToRTS() {
  ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 8;  // 1.048 ms
  // attr.timeout = 19;   // 2147 ms
  // attr.retry_cnt = 0;  // no retry
  // attr.rnr_retry = 0;  // no retry
  attr.retry_cnt = 7;  // infinite retry
  attr.rnr_retry = 7;  // infinite retry
  attr.max_rd_atomic = 1;

  int ret = ibv_modify_qp(qp_, &attr,
                          IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                              IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                              IBV_QP_MAX_QP_RD_ATOMIC);
  if (ret) die_perror("TransitQueuePairToRTS");
}

void Connection::RegisterMemory() {
  local_mr_ = ibv_reg_mr(pd_, local_buf_.data(), local_buf_.pool_size(),
                         IBV_ACCESS_LOCAL_WRITE);

  // Memory region exposed to remote machines
  if (memory_region_) {
    rdma_remote_mr_ =
        ibv_reg_mr(pd_, memory_region_->data(), memory_region_->size(),
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                       IBV_ACCESS_REMOTE_READ);
    if (!rdma_remote_mr_) die("ibv_reg_mr: rdma_remote_mr");
  }
}

void Connection::PostReceive() {
  ibv_recv_wr wr, *bad_wr = nullptr;
  ibv_sge sge;

  wr.wr_id = next_wr_id_.fetch_add(1);
  wr.next = nullptr;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  auto buf = local_buf_.Allocate();
  sge.addr = reinterpret_cast<uint64_t>(buf.data());
  sge.length = buf.size();
  sge.lkey = local_mr_->lkey;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    wr_ctx_.emplace(std::piecewise_construct, std::forward_as_tuple(wr.wr_id),
                    std::forward_as_tuple(std::move(buf)));
  }

  fprintf(stderr, "POST --> (RECV WR #%lu) [addr %lx, len %u, qp_num %u]\n",
          wr.wr_id, sge.addr, sge.length, qp_->qp_num);
  int ret = ibv_post_recv(qp_, &wr, &bad_wr);
  if (ret) die("ibv_post_recv");
}

void Connection::AsyncSend(OwnedMemoryBlock buf) {
  if (!is_connected_) die("Send: not connected.");
  ibv_send_wr wr, *bad_wr = nullptr;
  ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  wr.wr_id = next_wr_id_.fetch_add(1);
  wr.opcode = IBV_WR_SEND;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;

  auto msg = buf.AsMessageView();
  sge.addr = reinterpret_cast<uint64_t>(buf.data());
  sge.length = msg.total_length();
  sge.lkey = local_mr_->lkey;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    wr_ctx_.emplace(std::piecewise_construct, std::forward_as_tuple(wr.wr_id),
                    std::forward_as_tuple(std::move(buf)));
  }

  int ret = ibv_post_send(qp_, &wr, &bad_wr);
  if (ret) die("Connection::Send: ibv_post_send");
  fprintf(stderr, "POST --> (SEND WR #%lu) [addr %lx, len %u, qp_num %u]\n",
          wr.wr_id, sge.addr, sge.length, qp_->qp_num);
}

void Connection::AsyncRead(/* OwnedMemoryBlock buf, */ size_t offset,
                           size_t length) {
  ibv_send_wr wr, *bad_wr = nullptr;
  ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  wr.wr_id = next_wr_id_.fetch_add(1);
  wr.opcode = IBV_WR_RDMA_READ;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_mr_.addr + offset;
  wr.wr.rdma.rkey = remote_mr_.rkey;

  auto buf = local_buf_.Allocate();
  auto msg = buf.AsMessageView();
  sge.addr = reinterpret_cast<uintptr_t>(msg.bytes());
  sge.length = length;
  sge.lkey = local_mr_->lkey;
  msg.bytes_length() = length;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    wr_ctx_.emplace(std::piecewise_construct, std::forward_as_tuple(wr.wr_id),
                    std::forward_as_tuple(std::move(buf)));
  }

  int ret = ibv_post_send(qp_, &wr, &bad_wr);
  if (ret) die("Connection::PostRead: ibv_post_send");
  fprintf(stderr, "POST --> (READ WR #%lu) [offset %lx, len %lu, qp_num %u]\n",
          wr.wr_id, offset, length, qp_->qp_num);
}

void Connection::PollCompletionQueueBlocking() {
  constexpr int kPollTimeoutMills = 1;
  struct ibv_cq *cq;
  struct ibv_wc wc;
  void *ev_ctx;
  int ret;

  while (!poller_stop_) {
    do {
      ret = poll(&comp_channel_pollfd_, 1, kPollTimeoutMills);
    } while (ret == 0 && !poller_stop_);
    if (ret < 0) die("PollCompletionQueueBlocking: pool failed");
    if (poller_stop_) {
      break;
    }

    ret = ibv_get_cq_event(comp_channel_, &cq, &ev_ctx);
    if (ret < 0) {
      fprintf(stderr,
              "PollCompletionQueueBlocking: ibv_get_cq_event returns %d\n",
              ret);
      continue;
    }

    ibv_ack_cq_events(cq, 1);
    ret = ibv_req_notify_cq(cq, 0);
    if (ret) {
      fprintf(stderr, "ibv_req_notify_cq\n");
      continue;
    }
    while (!poller_stop_ && ibv_poll_cq(cq, 1, &wc)) {
      HandleWorkCompletion(&wc);
    }
  }
}

void Connection::PollCompletionQueueSpinning() {
  struct ibv_wc wc;
  while (!poller_stop_) {
    while (!poller_stop_ && ibv_poll_cq(cq_, 1, &wc)) {
      HandleWorkCompletion(&wc);
    }
    asm volatile("pause\n" : : : "memory");
  }
}

void Connection::HandleWorkCompletion(struct ibv_wc *wc) {
  if (wc->status != IBV_WC_SUCCESS) {
    fprintf(stderr, "COMPLETION FAILURE (%s WR #%lu) status[%d] = %s\n",
            (wc->opcode & IBV_WC_RECV) ? "RECV" : "SEND", wc->wr_id, wc->status,
            ibv_wc_status_str(wc->status));
    die("wc->status != IBV_WC_SUCCESS");
  }
  WorkRequestContext wr_ctx;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = wr_ctx_.find(wc->wr_id);
    if (iter == wr_ctx_.end()) {
      fprintf(stderr, "Cannot find context for wr_id #%lu\n", wc->wr_id);
      die("wc->wr_id not in wr_ctx_");
    }
    wr_ctx = std::move(iter->second);
  }
  if (wc->opcode & IBV_WC_RECV) {
    PostReceive();
    handler_->OnRecv(std::move(wr_ctx.buf));
    return;
  }
  switch (wc->opcode) {
    case IBV_WC_SEND: {
      handler_->OnSent(std::move(wr_ctx.buf));
      return;
    }
    case IBV_WC_RDMA_READ: {
      handler_->OnRdmaReadComplete(std::move(wr_ctx.buf));
      return;
    }
    // TODO: handle all opcode
    case IBV_WC_RDMA_WRITE:
    default:
      fprintf(stderr, "Unknown wc->opcode %d\n", wc->opcode);
      return;
  }
}

void Connection::MarkConnected() {
  do {
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    if (ibv_query_qp(qp_, &attr, IBV_QP_STATE, &init_attr)) {
      die("ibv_query_qp\n");
    }
    fprintf(stderr,
            "qp_state: %ld\n, cur_qp_state: %ld\n, path_mtu: %ld\n, "
            "path_mig_state: %ld\n, qkey: %ld\n, rq_psn: %ld\n, sq_psn: "
            "%ld\n, "
            "dest_qp_num: %ld\n, qp_access_flags: %ld\n, pkey_index: %ld\n, "
            "alt_pkey_index: %ld\n, en_sqd_async_notify: %ld\n, sq_draining: "
            "%ld\n, max_rd_atomic: %ld\n, max_dest_rd_atomic: %ld\n, "
            "min_rnr_timer: %ld\n, port_num: %ld\n, timeout: %ld\n, "
            "retry_cnt: "
            "%ld\n, rnr_retry: %ld\n, alt_port_num: %ld\n, alt_timeout: "
            "%ld\n",
            static_cast<int64_t>(attr.qp_state),
            static_cast<int64_t>(attr.cur_qp_state),
            static_cast<int64_t>(attr.path_mtu),
            static_cast<int64_t>(attr.path_mig_state),
            static_cast<int64_t>(attr.qkey), static_cast<int64_t>(attr.rq_psn),
            static_cast<int64_t>(attr.sq_psn),
            static_cast<int64_t>(attr.dest_qp_num),
            static_cast<int64_t>(attr.qp_access_flags),
            static_cast<int64_t>(attr.pkey_index),
            static_cast<int64_t>(attr.alt_pkey_index),
            static_cast<int64_t>(attr.en_sqd_async_notify),
            static_cast<int64_t>(attr.sq_draining),
            static_cast<int64_t>(attr.max_rd_atomic),
            static_cast<int64_t>(attr.max_dest_rd_atomic),
            static_cast<int64_t>(attr.min_rnr_timer),
            static_cast<int64_t>(attr.port_num),
            static_cast<int64_t>(attr.timeout),
            static_cast<int64_t>(attr.retry_cnt),
            static_cast<int64_t>(attr.rnr_retry),
            static_cast<int64_t>(attr.alt_port_num),
            static_cast<int64_t>(attr.alt_timeout));
    if (attr.qp_state != IBV_QPS_RTS) die("attr.qp_state != IBV_QPS_RTS");
  } while (false);

  if (poller_type_ == PollerType::kBlocking) {
    int ret = ibv_req_notify_cq(cq_, 0);
    if (ret) die("ibv_req_notify_cq");
    cq_poller_thread_ =
        std::thread(&Connection::PollCompletionQueueBlocking, this);
  } else {
    cq_poller_thread_ =
        std::thread(&Connection::PollCompletionQueueSpinning, this);
  }
  for (size_t i = 0; i < kRecvBacklog; ++i) {
    PostReceive();
  }

  is_connected_ = true;
}

}  // namespace ario
