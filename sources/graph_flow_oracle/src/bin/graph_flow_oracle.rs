use ig_graph_flow_oracle::{bits,carrier,token};
use imasm_core::{classic::Token,check::Graph,flow::flow_values};
use serde_json::{json,Value};

fn main(){
    let args:Vec<String>=std::env::args().collect();
    if args.len()==2 && args[1]=="--table" {
        let mut table=Vec::new();
        for code in 0..=Token::Rotat as u8 {
            let t=token(code).expect("bound discriminant");
            assert_eq!(Token::parse(t.name()),Some(t));
            let (input,output)=t.arity();
            table.push(json!({"id":code,"name":t.name(),"glyph":t.code(),"input_ports":input,"output_ports":output}));
        }
        assert_eq!(table.len(),Token::parse_all_names().as_array().unwrap().len());
        println!("{}",json!({"engine":"imasm_core::flow","tokens":table}));return;
    }
    if args.len()!=3 || args[1]!="--graph" {eprintln!("usage: graph_flow_oracle --table | --graph graph.json");std::process::exit(2);}
    let source=std::fs::read_to_string(&args[2]).expect("graph source");
    let input:Value=serde_json::from_str(&source).expect("graph JSON");let mut graph=Graph::new();
    for node in input["nodes"].as_array().expect("nodes") {
        graph.add(Token::parse(node.as_str().expect("opcode name")).expect("canonical opcode"));
    }
    for edge in input["edges"].as_array().expect("edges") {
        let a=edge[0].as_u64().expect("source index") as usize;
        let b=edge[1].as_u64().expect("destination index") as usize;
        assert!(a<graph.nodes.len()&&b<graph.nodes.len(),"edge boundary");graph.connect(a,b);
    }
    let seed=input["seed"].as_u64().expect("seed mask");assert!(seed<16);
    let readout=input["readout"].as_u64().expect("readout index") as usize;assert!(readout<graph.nodes.len());
    let grammar=graph.validate();
    if !grammar.is_empty(){println!("{}",json!({"engine":"imasm_core::flow","grammar":grammar,"executed":false}));return;}
    let (nodes,edges)=flow_values(&graph,carrier(seed as u8));
    let returned=bits(nodes[readout]);
    println!("{}",json!({"engine":"imasm_core::flow","grammar":grammar,"executed":true,
        "node_inputs":nodes.into_iter().map(bits).collect::<Vec<_>>(),
        "edge_values":edges.into_iter().map(bits).collect::<Vec<_>>(),
        "readout":returned,
        "closure":format!("{:?}",graph.closure_state()),"circuit_rank":graph.circuit_rank()}));
}
