/* A retained ported graph. One learned operator is shared at every edge action.
 * mode 0: learned propagation; 1: source oracle; 2: checked propagation with
 * preserved proposals; 3: learned propagation after optional correction.
 */
extern "C" unsigned int ig_flow_gate(unsigned int,unsigned int,unsigned int,unsigned int,unsigned int);
#define GRAPH_MAX_NODES 256
#define GRAPH_MAX_EDGES 768
static long graph_nodes,graph_edges,graph_readout,graph_seed,graph_mode;
static long graph_op[GRAPH_MAX_NODES],graph_in[GRAPH_MAX_NODES],graph_outdegree[GRAPH_MAX_NODES];
static long graph_from[GRAPH_MAX_EDGES],graph_to[GRAPH_MAX_EDGES],graph_slot[GRAPH_MAX_EDGES],graph_input_slot[GRAPH_MAX_EDGES];
static long graph_value[GRAPH_MAX_EDGES],graph_before[GRAPH_MAX_EDGES],graph_after[GRAPH_MAX_EDGES],graph_expected[GRAPH_MAX_EDGES];
static long graph_diagnostics[14],graph_started,graph_total_passes;
extern "C" long vgi_graph_reset(long nodes,long edges,long readout){
    if(nodes<1||nodes>GRAPH_MAX_NODES||edges<0||edges>GRAPH_MAX_EDGES||readout<0||readout>=nodes)return -80;
    graph_nodes=nodes;graph_edges=edges;graph_readout=readout;graph_started=graph_total_passes=0;
    for(long i=0;i<nodes;i++){graph_op[i]=-1;graph_in[i]=graph_outdegree[i]=0;}
    for(long i=0;i<edges;i++){graph_from[i]=graph_to[i]=graph_slot[i]=graph_input_slot[i]=-1;
        graph_value[i]=graph_before[i]=graph_after[i]=graph_expected[i]=0;}
    for(int i=0;i<14;i++)graph_diagnostics[i]=0;return 0;
}
extern "C" long vgi_graph_node(long index,long opcode){
    if(index<0||index>=graph_nodes||opcode<0||opcode>=GRAPH_TOKEN_COUNT)return -81;
    graph_op[index]=opcode;return 0;
}
extern "C" long vgi_graph_edge(long index,long from,long to,long outport,long inport){
    if(index<0||index>=graph_edges||from<0||from>=graph_nodes||to<0||to>=graph_nodes||outport<0||outport>2||inport<0||inport>2)return -82;
    graph_from[index]=from;graph_to[index]=to;graph_slot[index]=outport;graph_input_slot[index]=inport;return 0;
}
static long graph_validate(){
    long incoming[GRAPH_MAX_NODES]={},outgoing[GRAPH_MAX_NODES]={};
    for(long node=0;node<graph_nodes;node++)if(graph_op[node]<0)return -83;
    for(long edge=0;edge<graph_edges;edge++){
        long from=graph_from[edge],to=graph_to[edge];if(from<0||to<0)return -83;
        if(graph_slot[edge]>=graph_arity_out[graph_op[from]]||graph_input_slot[edge]>=graph_arity_in[graph_op[to]])return -84;
        incoming[to]++;outgoing[from]++;
        for(long previous=0;previous<edge;previous++)if((graph_from[previous]==from&&graph_slot[previous]==graph_slot[edge])||
            (graph_to[previous]==to&&graph_input_slot[previous]==graph_input_slot[edge]))return -84;
    }
    for(long node=0;node<graph_nodes;node++){
        if(incoming[node]>graph_arity_in[graph_op[node]]||outgoing[node]>graph_arity_out[graph_op[node]])return -84;
        graph_outdegree[node]=outgoing[node];
    }
    // Bind port ordinals to the active fan used by the source graph engine.
    for(long edge=0;edge<graph_edges;edge++)if(graph_slot[edge]>=graph_outdegree[graph_from[edge]])return -84;
    return 0;
}
static long graph_neural(long opcode,long input,long seed,long slot,long fan,long target,long steps){
    unsigned short query[8]={256,71,(unsigned short)opcode,(unsigned short)(266+input),(unsigned short)(266+seed),
                             (unsigned short)fan,(unsigned short)slot,257};
    for(int i=0;i<8;i++)retained_bank[i]=query[i];
    long loss=vgi_retained_update(8,0,2,target,steps,1);if(loss<0)return loss;
    long value=0;for(int k=0;k<4;k++)value|=((k%2?retained_first_f[259+k/2]:retained_first_t[259+k/2])>0)<<k;
    return value;
}
extern "C" long vgi_graph_observe(long kind,long index){
    if(kind==0&&index>=0&&index<14)return graph_diagnostics[index];
    if(kind==1&&index>=0&&index<graph_nodes)return graph_in[index];
    if(index<0||index>=graph_edges)return -85;
    if(kind==2)return graph_value[index];if(kind==3)return graph_before[index];
    if(kind==4)return graph_after[index];if(kind==5)return graph_expected[index];return -85;
}
static long graph_advance(long seed,long mode,long passes,long correction_steps,long readout,long restart){
    if(seed<0||seed>15||mode<0||mode>3||passes<0||passes>4096||correction_steps<0||correction_steps>100)return -86;
    long admission=graph_validate();if(admission<0)return admission;
    if(!parameters_initialized)vgi_retained_reset();
    graph_seed=seed;graph_mode=mode;
    if(restart){
        for(long e=0;e<graph_edges;e++)graph_value[e]=graph_before[e]=graph_after[e]=graph_expected[e]=0;
        graph_total_passes=0;
    }
    graph_started=1;
    for(int i=0;i<14;i++)graph_diagnostics[i]=0;
    long limit=passes?passes:4*(graph_edges?graph_edges:1)+4;
    graph_diagnostics[6]=mode;graph_diagnostics[7]=seed;graph_diagnostics[8]=graph_readout;graph_diagnostics[9]=limit;
    for(long pass=0;pass<limit;pass++){
        long changed=0,updates=0;
        for(long node=0;node<graph_nodes;node++){
            long input=0;for(long e=0;e<graph_edges;e++)if(graph_to[e]==node)input|=graph_value[e];
            graph_in[node]=input;
            for(long e=0;e<graph_edges;e++)if(graph_from[e]==node){
                long expected=(long)ig_flow_gate(graph_op[node],input,seed,graph_slot[e],graph_outdegree[node]);
                if(expected<0||expected>15)return -87;
                long before=expected,after=expected;
                if(mode!=1){
                    before=graph_neural(graph_op[node],input,seed,graph_slot[e],graph_outdegree[node],expected,0);
                    if(before<0)return before;after=before;
                    if((mode==2||mode==3)&&before!=expected&&correction_steps){
                        long loss=graph_neural(graph_op[node],input,seed,graph_slot[e],graph_outdegree[node],expected,correction_steps);
                        if(loss<0)return loss;
                        after=graph_neural(graph_op[node],input,seed,graph_slot[e],graph_outdegree[node],expected,0);
                        if(after<0)return after;updates+=correction_steps;graph_diagnostics[5]+=correction_steps;
                    }
                }
                graph_before[e]=before;graph_after[e]=after;graph_expected[e]=expected;
                graph_diagnostics[2]++;graph_diagnostics[3]+=before!=expected;graph_diagnostics[4]+=after!=expected;
                long value=(mode==1||mode==2)?expected:after;
                if(graph_value[e]!=value){graph_value[e]=value;changed=1;}
            }
        }
        graph_diagnostics[1]=pass+1;
        if(!changed){graph_diagnostics[0]=1;graph_diagnostics[10]=updates;break;}
    }
    graph_total_passes+=graph_diagnostics[1];graph_diagnostics[12]=graph_total_passes;
    graph_diagnostics[13]=parameter_digest();graph_diagnostics[11]=graph_in[graph_readout];
    if(readout==0)return graph_in[graph_readout];
    if(readout==1)return graph_diagnostics[0]|(graph_in[graph_readout]<<1);
    if(readout>=10&&readout<24)return graph_diagnostics[readout-10];return -88;
}
extern "C" long vgi_graph_run(long seed,long mode,long passes,long steps,long readout){
    return graph_advance(seed,mode,passes,steps,readout,1);
}
extern "C" long vgi_graph_continue(long mode,long passes,long steps,long readout){
    if(!graph_started)return -90;
    return graph_advance(graph_seed,mode,passes,steps,readout,0);
}
static long graph_write_state(){
    const unsigned char magic[8]={'G','G','S','T','A','T','E','1'};
    long header[7]={graph_nodes,graph_edges,graph_readout,graph_seed,graph_mode,graph_started,14};
    if(checkpoint_write(magic,8)<0||checkpoint_write(header,sizeof(header))<0||checkpoint_write(graph_diagnostics,sizeof(graph_diagnostics))<0)return -91;
    long*nodes[2]={graph_op,graph_in};long*edges[8]={graph_from,graph_to,graph_slot,graph_input_slot,graph_value,graph_before,graph_after,graph_expected};
    for(int i=0;i<2;i++)if(checkpoint_write(nodes[i],8*graph_nodes)<0)return -91;
    for(int i=0;i<8;i++)if(checkpoint_write(edges[i],8*graph_edges)<0)return -91;
    return 0;
}
extern "C" long vgi_graph_save(){
    if(vgi_write_current()<0||retained_write_state()<0||graph_write_state()<0)return -91;return 0;
}
/* GGRAPH1\0; u16 nodes,edges,readout,seed; node IDs; (from,to,outport,inport). */
extern "C" long vgi_graph_stream(long mode,long steps,long passes,long readout){
    if(!parameters_initialized)vgi_retained_reset();
    unsigned char header[16];if(stream_read(header,16)!=16)return -89;
    const unsigned char magic[8]={'G','G','R','A','P','H','1',0};
    const unsigned char restore_magic[8]={'G','G','R','S','T','R','1',0};
    int restore=1,ordinary=1;for(int i=0;i<8;i++){if(header[i]!=magic[i])ordinary=0;if(header[i]!=restore_magic[i])restore=0;}
    if(!restore&&!ordinary)return -89;
    long nodes=u16(header+8),edges=u16(header+10),sink=u16(header+12),seed=u16(header+14);
    long result=vgi_graph_reset(nodes,edges,sink);if(result<0)return result;
    unsigned char row[8];
    for(long i=0;i<nodes;i++){if(stream_read(row,2)!=2)return -89;result=vgi_graph_node(i,u16(row));if(result<0)return result;}
    for(long i=0;i<edges;i++){
        if(stream_read(row,8)!=8)return -89;
        result=vgi_graph_edge(i,u16(row),u16(row+2),u16(row+4),u16(row+6));if(result<0)return result;
    }
    if(restore){
        unsigned long total;long diagnostics[14];
        if(stream_read(&total,8)!=8||total>0x7fffffffffffffffUL||stream_read(diagnostics,sizeof(diagnostics))!=(long)sizeof(diagnostics))return -94;
        for(long i=0;i<nodes;i++){if(stream_read(row,2)!=2||u16(row)>15)return -94;graph_in[i]=u16(row);}
        for(long i=0;i<edges;i++){
            if(stream_read(row,8)!=8)return -94;
            for(int k=0;k<4;k++)if(u16(row+2*k)>15)return -94;
            graph_value[i]=u16(row);graph_before[i]=u16(row+2);graph_after[i]=u16(row+4);graph_expected[i]=u16(row+6);
        }
        if(retained_read_state(0,0,1)<0)return -94;
        graph_seed=seed;graph_mode=mode;graph_started=1;graph_total_passes=total;
        for(int i=0;i<14;i++)graph_diagnostics[i]=diagnostics[i];
    }
    unsigned char extra;if(stream_read(&extra,1)!=0)return -89;
    return restore?vgi_graph_continue(mode,passes,steps,readout):vgi_graph_run(seed,mode,passes,steps,readout);
}
static long retained_graph_local_validate(const unsigned short*query){
    if(query[0]!=256||query[1]!=71||query[2]>=GRAPH_TOKEN_COUNT||query[3]<266||query[3]>281||
       query[4]<266||query[4]>281||query[5]<1||query[5]>3||query[6]>=query[5]||query[7]!=257)return -92;
    if(query[5]>graph_arity_out[query[2]])return -92;return 0;
}
static long retained_graph_local_record(long steps,long record){
    if(retained_graph_local_validate(retained_bank)<0)return -92;
    unsigned short query[8];for(int i=0;i<8;i++)query[i]=retained_bank[i];
    long opcode=query[2],input=query[3]-266,seed=query[4]-266,fan=query[5],slot=query[6];
    long expected=(long)ig_flow_gate(opcode,input,seed,slot,fan);if(expected<0||expected>15)return -87;
    unsigned long previous=retained_records;
    long before_loss=vgi_retained_update(8,0,2,expected,0,1);if(before_loss<0)return before_loss;
    long before=0;for(int k=0;k<4;k++)before|=((k%2?retained_first_f[259+k/2]:retained_first_t[259+k/2])>0)<<k;
    long applied=0;if(before!=expected&&steps){
        long loss=vgi_retained_update(8,0,2,expected,steps,1);if(loss<0)return loss;applied=steps;
    }
    long after_loss=vgi_retained_update(8,0,2,expected,0,1);if(after_loss<0)return after_loss;
    long after=0;for(int k=0;k<4;k++)after|=((k%2?retained_first_f[259+k/2]:retained_first_t[259+k/2])>0)<<k;
    long fields[13]={expected,before,after,before^expected,after^expected,-1,-1,before,after,after==expected,applied,before_loss,after_loss};
    for(int i=0;i<13;i++)retained_oracle_trace[i]=fields[i];retained_oracle_log(record);
    retained_records=previous+1;retained_kind=4;retained_bits=expected;retained_committed=applied;retained_requested=steps;retained_complete=1;
    return after_loss;
}
