"""The launch gate rejects plausible-looking but incompatible evaluation evidence."""
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from empowerment_warp.evaluation_checks import check_recipe, check_evaluation, check_evidence, source_hashes
from empowerment_warp.evaluation_protocol import evaluation_keys
from empowerment_warp.recipes import resolve

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def recipe():
    # The evidence artifacts below omit the switch-memory key, which means reset memory.
    return resolve(ROOT/'configs/craftax_classic/see_reset_memory.yaml')[0]


def artifact(directory,recipe,step):
    directory.mkdir(parents=True,exist_ok=True)
    keys=np.asarray(evaluation_keys('craftax:classic-symbolic',0,step))
    skills=jax.random.normal(jnp.asarray(keys[2]),(128,16))
    skills/=jnp.sqrt(jnp.square(skills).sum(-1,keepdims=True)+1e-8)
    prefix=np.zeros((128,22),bool);prefix[:,0]=True
    final=prefix.copy();final[:,1]=True
    done=np.zeros((1000,128),bool);done[-1]=True
    on=np.ones((1000,128),bool);on[:100]=False
    data=dict(valid=np.ones((1001,128),bool),dones=done,skill_on=on,skills=np.array(skills),
              reset_keys=np.asarray(jax.random.split(jnp.asarray(keys[0]),128)),
              initial_achievements=np.zeros((128,22),bool),prefix_achievements=prefix,final_achievements=final,
              total_achievement_counts=np.full(128,2),potential_achievement_counts=np.ones(128),
              skill_achievement_counts=np.ones(128),episode_lengths=np.full(128,1000))
    np.savez_compressed(directory/'trace.npz',**data)
    row={'env_steps':step,'checkpoint_step':step,'learner_updates':step//131072,
         'evaluation/achievement_count':2.,'evaluation/potential_policy/achievement_count':1.,
         'evaluation/skill_policy/new_achievement_count':1.,'evaluation/completed_episodes':128.,
         'protocol':dict(name='craftax-128-fresh-switching-v2',engine='pqn',environment='craftax:classic-symbolic',
             num_envs=128,prefix_steps=100,episode_length=1000,contexts=128,skills_per_context=1,
             sampling='fresh_prior_and_resets_per_checkpoint',backend='gpu',deterministic=False,
             seed=0,rng_keys=keys.tolist(),evaluation_source=source_hashes('pqn','craftax:classic-symbolic'),
             trace=str(directory/'trace.npz'))}
    path=directory/'evaluation.json';path.write_text(json.dumps(row))
    return path,row,data


def test_ablation_recipes_resolve_to_contract():
    for env in ('ant','ant_soccer','builderbench_corner3','craftax_classic'):
        for arm in ('see','no_switch','pair_decoder','lambda0','lambda1',
                    'joint_mi' if env=='craftax_classic' else 'cmi'):
            check_recipe(resolve(ROOT/'configs'/env/(arm+'.yaml'))[0])
        if env != 'craftax_classic':
            for arm in ('latentgce','vim'):
                check_recipe(resolve(ROOT/'configs'/env/(arm+'.yaml'))[0])

@pytest.mark.parametrize('key,value',[('eval_contexts',32),('eval_skills_per_context',4),
    ('eval_every_segments',100),('eval_media_interval',10_000_000),('eval_episode_length',400),
    ('eval_at_start',False),('discrete_skills',True)])
def test_wrong_craftax_recipe_rejected(recipe,key,value):
    recipe['settings'][key]=value
    with pytest.raises(ValueError,match='evaluation contract'):
        check_recipe(recipe)


def test_missing_field_and_wrong_continuous_preset_rejected(recipe):
    del recipe['settings']['eval_contexts']
    with pytest.raises(ValueError,match='missing required field'):
        check_recipe(recipe)
    ant=resolve(ROOT/'configs/ant/see.yaml')[0]
    ant['settings']['antsoccer_preset']='standard'
    with pytest.raises(ValueError,match='reset preset'):
        check_recipe(ant)


def test_phase_arrays_and_current_source_pass(tmp_path,recipe):
    path,_,_=artifact(tmp_path,recipe,0)
    assert check_evaluation(path,recipe)['target']==0


def test_evidence_must_match_switch_memory_flag(tmp_path, recipe):
    path, row, _ = artifact(tmp_path, recipe, 0)
    recipe['settings']['pqn_preserve_memory_at_switch'] = True
    with pytest.raises(ValueError, match='switch memory'):
        check_evaluation(path, recipe)
    row['protocol']['pqn_preserve_memory_at_switch'] = True
    path.write_text(json.dumps(row))
    assert check_evaluation(path, recipe)['target'] == 0

@pytest.mark.parametrize('change,match',[
    ('source','evaluation source'),('version','protocol version'),('missing','missing required'),
    ('count','worlds'),('wrong_total','recomputation'),('fixed_skills','fresh prior skills'),
    ('restarted','first-episode masking'),('wrong_phase','potential phase commands'),
    ('trained_zero','pretraining checkpoint'),('updated_zero','pretraining learner')])
def test_corrupted_evidence_rejected(tmp_path,recipe,change,match):
    path,row,data=artifact(tmp_path,recipe,0)
    if change=='source':row['protocol']['evaluation_source']['training/pqn/branch_eval.py']='old-code'
    elif change=='version':row['protocol']['name']='old-protocol'
    elif change=='missing':del row['evaluation/skill_policy/new_achievement_count']
    elif change=='count':row['protocol']['num_envs']=1024
    elif change=='wrong_total':row['evaluation/achievement_count']=99
    elif change=='fixed_skills':data['skills'][:]=data['skills'][0]
    elif change=='restarted':data['valid'][5]=False
    elif change=='wrong_phase':data['skill_on'][:100]=True
    elif change=='trained_zero':row['checkpoint_step']=1000
    elif change=='updated_zero':row['learner_updates']=1
    path.write_text(json.dumps(row));np.savez_compressed(tmp_path/'trace.npz',**data)
    with pytest.raises(ValueError,match=match):check_evaluation(path,recipe)


def test_fixed_video_allowed_but_fixed_metric_skills_at_next_checkpoint_fail(tmp_path,recipe):
    p0,_,a0=artifact(tmp_path/'first',recipe,0)
    p1,row,a1=artifact(tmp_path/'later',recipe,10_092_544)
    video=tmp_path/'fixed-probes.mp4';video.write_bytes(b'unit-test-video-placeholder')
    row['figures']={'protocol':{'contributes_to_metrics':False},'saved_paths':{'video':str(video)}}
    p1.write_text(json.dumps(row))
    manifest={'evaluations':[str(p0),str(p1)],'media':[{'path':str(video),'role':'fixed_diagnostic',
              'evaluation':str(p1),
              'sha256':hashlib.sha256(video.read_bytes()).hexdigest()}]}
    m=tmp_path/'evidence.json';m.write_text(json.dumps(manifest))
    assert check_evidence(m,recipe)['status']=='passed'
    a1['skills']=a0['skills'];np.savez_compressed(p1.parent/'trace.npz',**a1)
    with pytest.raises(ValueError,match='fresh prior skills'):check_evidence(m,recipe)


def test_full_submission_stops_before_any_subprocess_without_evidence(monkeypatch,tmp_path):
    from empowerment_warp import submit
    monkeypatch.setattr(submit.subprocess,'run',lambda *a,**k:pytest.fail('must fail before shell or sbatch'))
    with pytest.raises(SystemExit):
        submit.main(['--project-root',str(ROOT),'--config',str(ROOT/'configs/craftax_classic/see.yaml'),
                     '--run-root',str(tmp_path/'never-created'),'--mem','16G','--time','01:00:00'])
    assert not (tmp_path/'never-created').exists()


@pytest.mark.parametrize('change,match',[
    ('missing_zero','pretraining'),('different_seed','share the training seed'),
    ('missing_media','missing review media'),('hash','media hash'),
    ('unrelated_video','video not in evaluation media manifest'),
    ('diagnostic_scoring','diagnostic metric contribution')])
def test_bad_manifest_rejected(tmp_path,recipe,change,match):
    p0,_,_=artifact(tmp_path/'first',recipe,0)
    p1,row,_=artifact(tmp_path/'later',recipe,10_092_544)
    video=tmp_path/'video.mp4';video.write_bytes(b'unit-test-video-placeholder')
    row['figures']={'protocol':{'contributes_to_metrics':False},'saved_paths':{'video':str(video)}}
    manifest={'evaluations':[str(p0),str(p1)],'media':[{'path':str(video),'role':'fixed_diagnostic',
        'evaluation':str(p1),'sha256':hashlib.sha256(video.read_bytes()).hexdigest()}]}
    if change=='missing_zero':
        p2,_,_=artifact(tmp_path/'last',recipe,20_185_088)
        manifest['evaluations']=[str(p1),str(p2)]
    elif change=='different_seed':
        # Use a separately valid seed-1 record, rather than only changing its label.
        row['protocol']['seed']=1
        keys=np.asarray(evaluation_keys('craftax:classic-symbolic',1,10_092_544))
        row['protocol']['rng_keys']=keys.tolist()
        with np.load(p1.parent/'trace.npz') as archive:
            data=dict(archive)
        z=jax.random.normal(jnp.asarray(keys[2]),(128,16))
        data['skills']=np.asarray(z/jnp.sqrt(jnp.square(z).sum(-1,keepdims=True)+1e-8))
        data['reset_keys']=np.asarray(jax.random.split(jnp.asarray(keys[0]),128))
        np.savez_compressed(p1.parent/'trace.npz',**data)
    elif change=='missing_media':manifest['media']=[]
    elif change=='hash':manifest['media'][0]['sha256']='wrong'
    elif change=='unrelated_video':row['figures']['saved_paths']={}
    elif change=='diagnostic_scoring':row['figures']['protocol']['contributes_to_metrics']=True
    p1.write_text(json.dumps(row))
    path=tmp_path/'evidence.json';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match=match):check_evidence(path,recipe)
