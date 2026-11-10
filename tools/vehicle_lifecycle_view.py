"""Local vehicle-parent review; no mutation of inference, labels or scores."""
import html
import re


def group_vehicle_cards(page,document):
    """Reorganize evidence without changing identity assignments or dropping crops."""
    esc=lambda value:html.escape(str(value),quote=True)
    pattern=r'<section id="(PT\d+)">.*?</section>'
    matches=list(re.finditer(pattern,page,re.S))
    cards={match.group(1):match.group(0) for match in matches}
    expected={g['id'] for g in document['groups']}
    if set(cards)!=expected or len(matches)!=len(cards):raise ValueError('Every plate history needs exactly one original evidence card')
    page=re.sub(pattern,'',page,flags=re.S)
    vehicles={v['id']:v for v in document['vehicles']};results={v['id']:v for v in document['vehicle_results']}
    attached={};unassigned=[]
    for group in document['groups']:
        parent=group.get('parent_vehicle_id')
        if parent in vehicles:attached.setdefault(parent,[]).append(group['id'])
        else:unassigned.append(group['id'])
    parent_cards=[]
    for parent,children in attached.items():
        result=results.get(parent);reading=result['methods']['top'] if result else None
        heading=esc(parent)+' · '+esc(vehicles[parent]['status'])+' · '+str(len(children))+' plate histories · '+esc(reading or 'No combined reading')
        parent_cards.append('<details class="vehicle-card" id="'+esc(parent)+'"><summary>'+heading+'</summary><p>Confirmed ownership groups these views. PT labels describe evidence histories, not additional vehicles.</p>'+''.join(cards[c] for c in children)+'</details>')
    body='<section id="vehicle-evidence"><h2>Vehicles with confirmed plate ownership</h2><p>'+str(len(attached))+' vehicle histories contain assigned plate views; '+str(len(unassigned))+' plate histories remain unassigned. These are histories, not verified unique vehicle counts.</p>'+''.join(parent_cards)+'</section>'
    body+='<details id="unassigned-plate-views"><summary>Review '+str(len(unassigned))+' unassigned plate histories</summary><p>Ownership is unresolved. These crops remain available and have not been silently merged.</p>'+''.join(cards[c] for c in unassigned)+'</details>'
    body+='''<style>.vehicle-card,#unassigned-plate-views{padding:14px;margin:12px 0;border:1px solid #cbd9ce;border-radius:10px}summary{cursor:pointer;font-weight:650}section[id]{scroll-margin-top:20px}</style>
<script>(()=>{function reveal(){let id;try{id=decodeURIComponent(location.hash.slice(1));}catch{return;}const target=document.getElementById(id);for(let node=target;node;node=node.parentElement){if(node.tagName==='DETAILS')node.open=true;}}window.addEventListener('hashchange',reveal);reveal();})();</script>'''
    marker='<section><h2>All 18 development references</h2>'
    return page.replace(marker,body+marker,1)


def case_panel(summary):
    esc=lambda value:html.escape(str(value),quote=True)
    rows=[]
    for case in summary['reported_case_checks']:
        descriptions=[]
        for previous,mapped in case['mapping'].items():
            destinations=', '.join('<a href="#'+esc(m['plate_history'])+'">'+esc(m['plate_history'])+'</a> / '+esc(m['vehicle'] or 'unassigned vehicle') for m in mapped) or 'No uniquely matched crop'
            descriptions.append(esc(previous)+' → '+destinations)
        rows.append('<tr><td>'+', '.join(esc(v) for v in case['previous_cluster'])+'</td><td>'+('<br>'.join(descriptions))+
                    '</td><td>'+('One confirmed vehicle' if case['single_confirmed_parent'] else 'Still partly unresolved')+'</td></tr>')
    return '<section id="reported-cases"><h2>Your reported examples</h2><p>Previous run IDs are mapped by source time and crop geometry after inference. These notes never guide detection or OCR.</p><table><tr><th>Previous IDs</th><th>New plate / vehicle histories</th><th>Review state</th></tr>'+''.join(rows)+'</table><p>Previous PT0023 is a different autorickshaw from the car in PT0004/PT0006/PT0007. Unassigned ownership is shown explicitly.</p></section>'


def augment(page,document,report=None):
    esc=lambda value:html.escape(str(value),quote=True)
    results={r['id']:r for r in document['vehicle_results']}
    children={}
    for group in document['groups']:
        if group.get('parent_vehicle_id'):children.setdefault(group['parent_vehicle_id'],[]).append(group['id'])
    rows=[]
    for vehicle in document['vehicles']:
        result=results.get(vehicle['id']);linked=children.get(vehicle['id'],[])
        links=', '.join('<a href="#'+esc(g)+'">'+esc(g)+'</a>' for g in linked) or 'No confirmed plate view'
        reading=esc(result['methods']['top'] or 'No reading') if result else 'No confirmed OCR evidence'
        gate='Conflicting plate views' if result and result['conflicting_plate_views'] else (
            'Text gate passed; inspect evidence' if result and result['methods']['accepted'] else 'Unconfirmed')
        rows.append('<tr><td>'+esc(vehicle['id'])+'</td><td>'+esc(vehicle['status'])+'</td><td>'+esc(vehicle.get('close_reason',''))+
                    '</td><td>'+esc(vehicle['observation_count'])+'</td><td>'+links+'</td><td>'+reading+'</td><td>'+gate+'</td></tr>')
    metrics=''
    if report:
        text=report['summary']['multi'];owned=report['vehicle_summary']
        metrics='<p><strong>Plate text: '+esc(text['exact'])+'/'+esc(text['targets'])+' selected references. Confirmed-vehicle readings: '+esc(owned['exact'])+'/'+esc(owned['targets'])+'.</strong> Text correctness and vehicle ownership are evaluated separately; unmatched targets remain misses.</p>'
    panel='''<section id="vehicle-lifecycle"><h2>Persistent vehicle histories</h2>'''+metrics+'''
<p>A temporary detector miss keeps a predicted position for up to one second, followed by an unresolved history for up to five seconds. Confirmed boundary exits, camera resets and unresolved expiry are shown separately. Predictions supply no OCR votes.</p>
<p>Several plate views may belong to one vehicle. Only confirmed ownership contributes to its combined reading; one timestamp supplies at most one OCR observation. Vehicle and plate IDs are specific to this run.</p>
<div style="position:relative;background:#132b29"><video id="lifecycle-video" controls preload="metadata" style="display:block;width:100%" src="http://127.0.0.1:7798/video"></video><canvas id="lifecycle-overlay" style="position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none"></canvas></div>
<p id="lifecycle-state" aria-live="polite">Loading completed analysis…</p><p>Solid teal: observed vehicle. Dashed amber: predicted position. Dormant histories have no drawn box. This is a replay of completed analysis.</p>
<details><summary>Inspect all '''+str(len(document['vehicles']))+''' vehicle history records</summary><div class="scroll"><table><tr><th>Vehicle</th><th>Final state</th><th>Closing reason</th><th>Actual observations</th><th>Plate histories</th><th>Combined hypothesis</th><th>Text gate</th></tr>'''+''.join(rows)+'''</table></div></details></section>
<script>
(() => {
 const video=document.querySelector('#lifecycle-video'), canvas=document.querySelector('#lifecycle-overlay'), label=document.querySelector('#lifecycle-state');
 const context=canvas.getContext('2d'); let timeline=null;
 function draw(seconds=video.currentTime) {
  if(!timeline)return;
  const frames=timeline.frames, ms=seconds*1000; let lo=0,hi=frames.length;
  while(lo<hi){const m=(lo+hi)>>1;if(frames[m].t<ms)lo=m+1;else hi=m;}
  let index=Math.min(lo,frames.length-1);if(index>0&&Math.abs(frames[index-1].t-ms)<=Math.abs(frames[index].t-ms))index--;
  const frame=frames[index]; context.clearRect(0,0,canvas.width,canvas.height);
  for(const track of frame.tracks){
   if(!track.box)continue; const [x1,y1,x2,y2]=track.box;
   context.strokeStyle=track.observed?'#14b8a6':'#f5b942';context.lineWidth=5;context.setLineDash(track.observed?[]:[20,12]);context.strokeRect(x1,y1,x2-x1,y2-y1);
   const kind=({2:'car',3:'motorcycle',5:'bus',7:'truck'})[track.vehicle_class]||'vehicle';
   context.font='34px system-ui';context.fillStyle=context.strokeStyle;context.fillText(track.id+' '+kind+' · '+track.status,Math.max(0,x1),Math.max(38,y1-8));
  }
  const count=status=>frame.tracks.filter(t=>t.status===status).length;
  label.textContent=(frame.t/1000).toFixed(3)+' s · '+count('visible')+' observed · '+count('predicted')+' predicted · '+count('lost_unresolved')+' unresolved';
 }
 function tick(_,metadata){draw(metadata.mediaTime);video.requestVideoFrameCallback(tick);}
 video.addEventListener('seeked',()=>draw());video.addEventListener('loadedmetadata',()=>draw());
 video.addEventListener('error',()=>{label.textContent='Video unavailable. Open the local tracking viewer on port 7798.';});
 fetch('vehicle-timeline.json').then(r=>{if(!r.ok)throw Error('Timeline unavailable');return r.json();}).then(data=>{
  timeline=data;canvas.width=data.width;canvas.height=data.height;draw();
  if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(tick);else video.addEventListener('timeupdate',()=>draw());
 }).catch(e=>{label.textContent=e.message;});
})();
</script>'''
    page=page.replace('Motion-supported plate recognition','Vehicle-led plate recognition')
    marker='<section><h2>All 18 development references</h2>'
    return group_vehicle_cards(page.replace(marker,panel+marker,1),document)
