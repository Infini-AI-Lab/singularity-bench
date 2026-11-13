"""Launch pinned Miles standalone TITO sessions, without the Miles training driver."""
import argparse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--backend-url',required=True)
    p.add_argument('--instance-id',required=True);p.add_argument('--host',default='0.0.0.0')
    p.add_argument('--port',type=int,default=30001);p.add_argument('--debug-dir',required=True)
    a=p.parse_args()
    import uuid
    uuid.UUID(a.instance_id)
    from miles.rollout.session.config import SessionServerConfig
    from miles.rollout.session.server import run_session_server
    config=SessionServerConfig(host=a.host,port=a.port,instance_id=a.instance_id,
        backend_url=a.backend_url.rstrip('/'),timeout=900.0,hf_checkpoint=a.model,
        chat_template_path=None,tito_model='glm47',apply_chat_template_kwargs=None,
        use_rollout_routing_replay=False,use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,sglang_speculative_algorithm=None,
        num_layers=47,moe_router_topk=4,save_debug_trajectory_data=a.debug_dir,
        lora_rank=0,lora_adapter_path=None,lora_train_only=False,use_session_server=True,
        session_message_matcher='strict',pause_generation_mode=None,
        session_sample_picker_path=None,session_sample_postprocessor_path=None)
    run_session_server(config)

if __name__=='__main__':
    main()
