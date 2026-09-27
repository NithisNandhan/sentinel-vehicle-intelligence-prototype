"""Contract/security checks for the standalone engineering renderer."""
import shutil
import subprocess
import unittest

from sentinel_edge.engineering_page import engineering_page


class EngineeringPageTests(unittest.TestCase):
    def test_token_cannot_close_script_and_profile_is_escaped(self):
        page = engineering_page('</script><img src=x onerror=alert(1)>', 'general<script>')
        self.assertNotIn('</script><img', page)
        self.assertIn(r'\u003c/script\u003e', page)
        self.assertIn('general&lt;script&gt;', page)
        self.assertNotIn('general<script>', page)
        self.assertNotIn('__TOKEN__', page)
        self.assertNotIn('__PROFILE__', page)

    def test_placeholder_like_values_remain_literal_data(self):
        page = engineering_page('__PROFILE__', '__TOKEN__')
        self.assertIn('const viewerToken="__PROFILE__";', page)
        self.assertIn('id="model-profile">__TOKEN__</dd>', page)

    def test_sensitive_requests_require_unlock_and_no_persistent_storage(self):
        page = engineering_page('viewer-test-token', 'general')
        self.assertIn("headers['X-Operator-Key']=operatorKey", page)
        self.assertIn("'X-Viewer-Token':viewerToken", page)
        self.assertIn("request('/engineering')", page)
        self.assertIn("if(!operatorKey||busy||unlockBusy", page)
        self.assertIn("if(!operatorKey||catalogueBusy", page)
        self.assertIn("operatorKey=''", page)
        for prohibited in ('localStorage', 'sessionStorage', 'indexedDB', 'document.cookie',
                           'console.log', 'innerHTML', 'insertAdjacentHTML', 'eval('):
            self.assertNotIn(prohibited, page)
        self.assertIn("el.textContent=String(content)", page)
        self.assertIn('type="password" autocomplete="off"', page)
        self.assertIn("cache:'no-store'", page)

    def test_images_start_hidden_and_clear_on_error_stale_lock_and_switch(self):
        page = engineering_page('token')
        self.assertIn('id="inspection" hidden', page)
        self.assertIn('id="frame"', page)
        self.assertIn("element.removeAttribute('src')", page)
        self.assertIn("latestSnapshot=null;selectedTrack=null", page)
        self.assertIn("clearEvidence('Evidence unavailable')", page)
        self.assertIn("clearEvidence('Shared camera changed')", page)
        self.assertIn("age>5", page)
        self.assertIn("Date.now()-lastGoodEvidence+latestSnapshot.frame_age_seconds*1000>5000", page)
        self.assertIn("epoch!==generation||key!==operatorKey", page)
        self.assertIn("document.visibilityState!=='visible'", page)

    def test_shared_session_uses_heartbeat_not_reselection_on_close(self):
        page = engineering_page('token')
        self.assertIn("request('/heartbeat'", page)
        self.assertIn("body:JSON.stringify({camera_id:activeCamera})", page)
        self.assertNotIn('sendBeacon', page)
        pagehide = page.split("window.addEventListener('pagehide'", 1)[1].split('setInterval', 1)[0]
        self.assertNotIn("'/session'", pagehide)
        self.assertIn("operatorKey=''", pagehide)
        self.assertIn('does not stop a session another interface is using', page)

    def test_candidate_and_evidence_budgets_and_honest_reconstruction(self):
        page = engineering_page('token')
        self.assertIn('track.candidates.slice(0,200)', page)
        self.assertIn('expanded?200:20', page)
        self.assertIn('track.frames.slice(-8)', page)
        self.assertIn('Character scores are model outputs, not calibrated probabilities', page)
        self.assertIn("status:'not_run'", page)
        self.assertIn('Model acceptance is not human verification', page)
        self.assertIn('Full text is an unverified hypothesis', page)
        self.assertIn("['vehicle','plate','ocr','fusion']", page)
        self.assertIn('debug.original_frame', page)
        self.assertIn('scene displays use JPEG; native crops below use PNG', page)

    def test_standalone_page_has_no_remote_scripts_or_media(self):
        page = engineering_page('token')
        self.assertNotIn('<script src=', page)
        self.assertNotIn('<link ', page)
        self.assertNotIn('src="http', page)
        self.assertNotIn('download=', page)
        self.assertIn('rel="noopener noreferrer"', page)
        self.assertIn("'data:image/'+mime+';base64,'", page)

    def test_configured_relays_are_not_presented_as_full_catalogue(self):
        page = engineering_page('token')
        self.assertIn("result.scope==='configured_relays'", page)
        self.assertIn('Configured CAM01/CAM02 relays only; full catalogue unavailable', page)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is optional for renderer behaviour tests')
    def test_browser_behaviour_unlock_clear_safe_text_and_shared_heartbeat(self):
        # Exercise the shipped script with a minimal DOM; no external libraries,
        # real credentials, camera access, media files or server are required.
        script = engineering_page('viewer-test-token', 'general').split('<script>', 1)[1].split('</script>', 1)[0]
        environment = r'''
import assert from 'node:assert/strict';
class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.listeners={};this.attributes={};this.hidden=false;this.value='';this.style={};this._text='';this.className='';this.classList={toggle:()=>{}};}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>child.textContent).join('');}
  get options(){return this.children;}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=[...children];this._text='';}
  setAttribute(name,value){this.attributes[name]=String(value);}
  removeAttribute(name){delete this.attributes[name];if(name==='src')delete this.src;}
  addEventListener(name,fn){this.listeners[name]=fn;}
}
const elements=new Map(),timers=[],requests=[],documentEvents={};
globalThis.document={visibilityState:'visible',getElementById:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement:tag=>new Element(tag),addEventListener:(name,fn)=>{documentEvents[name]=fn;}};
globalThis.window={addEventListener:()=>{}};
globalThis.setInterval=(fn,ms)=>{timers.push({fn,ms});return timers.length;};
globalThis.setTimeout=()=>1;globalThis.clearTimeout=()=>{};
const fixture={status:'live',camera_id:'cam17',frame_age_seconds:.1,frame:'AA==',resolution:[1920,1080],pts_ms:4000,decoded_frames:20,analysed_frames:3,vehicle_boxes:2,plate_crops:3,maximum_distinct_observations:3,unique_model_accepted_unverified:0,model_profile:'general',debug:{original_frame:'AQ==',stage_ms:{vehicle:100,plate:20,ocr:30,fusion:4},roi_budget:3,tracks:[{track_id:'7',vehicle_box:[1,2,3,4],vehicle_class:'car',observations:3,recognition_state:'uncertain',masked_plate:'MH****37',seen_pts_ms:4000,candidates:[{plate:'<img src=x onerror=alert(1)>',score:.6,minimum_character_score:.4,log_score:-10,observations:3}],gate_reasons:['Uncertain character'],frames:[{image:'AA==',width:120,height:32,pts_ms:4000,quality:.8,text:'MH02EK0837',characters:[{character:'M',score:.8,alternatives:[{character:'N',score:.2}]}]}],reconstruction:{status:'not_run',reason:'Native OCR baseline'}}]}};
let deny=false;
globalThis.fetch=async(path,options)=>{requests.push({path,headers:{...options.headers},body:options.body});return{ok:!deny,status:deny?403:200,json:async()=>path==='/cameras'?{available:true,cameras:[{id:'cam17',name:'Test authorised camera'}]}:path==='/engineering'?structuredClone(fixture):{ok:true}};};
const flush=()=>new Promise(resolve=>setImmediate(resolve));
'''
        checks = r'''
await flush();assert.equal(requests.length,0,'Locked page must not request protected endpoints');
$('operator-key').value='test-operator-key';await $('unlock-form').listeners.submit({preventDefault:()=>{}});await flush();
assert.equal(operatorKey,'test-operator-key');assert.equal($('operator-key').value,'');assert.equal($('inspection').hidden,false);
assert.ok(requests.some(request=>request.path==='/engineering'&&request.headers['X-Operator-Key']==='test-operator-key'));
assert.equal(requests.filter(request=>request.path==='/session').length,0,'Unlock must not start a camera');
assert.equal($('frame').src,'data:image/jpeg;base64,AA==');
assert.equal($('candidate-body').children[0].children[1].textContent,'<img src=x onerror=alert(1)>');
assert.equal($('candidate-body').children[0].children[1].children.length,0,'Hypothesis must be text, never markup');
$('show-original').listeners.click();assert.equal($('frame').src,'data:image/jpeg;base64,AQ==');
$('camera-select').value='cam14';timers.find(timer=>timer.ms===20000).fn();await flush();
assert.equal(JSON.parse(requests.at(-1).body).camera_id,'cam17','Heartbeat must use active shared camera, not pending dropdown');
$('lock').listeners.click();assert.equal(operatorKey,'');assert.equal(latestSnapshot,null);assert.equal($('frame').src,undefined);assert.equal($('evidence-grid').children.length,0);assert.equal($('inspection').hidden,true);
const count=requests.length;await refresh();await catalogue();assert.equal(requests.length,count);
deny=true;$('operator-key').value='wrong-key';await $('unlock-form').listeners.submit({preventDefault:()=>{}});
assert.equal(operatorKey,'');assert.equal($('inspection').hidden,true);assert.equal(latestSnapshot,null);
deny=false;$('operator-key').value='test-operator-key';await $('unlock-form').listeners.submit({preventDefault:()=>{}});await flush();
assert.ok($('frame').src);render({...structuredClone(fixture),frame_age_seconds:6});assert.equal($('frame').src,undefined);assert.equal(latestSnapshot,null);assert.equal($('evidence-grid').children.length,0);
'''
        completed = subprocess.run(
            [shutil.which('node'), '--input-type=module'], input=environment + script + checks,
            text=True, capture_output=True, timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
