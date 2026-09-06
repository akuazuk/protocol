const {chromium}=require('playwright');
const fs=require('fs');
const path=require('path');
// Run from repository root with Playwright on NODE_PATH. Synthetic API only.
const root=process.cwd();
fs.mkdirSync('/private/tmp/mo-audit-artifacts',{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.addInitScript(()=>localStorage.setItem('protocol_methodist_token','synthetic-local-only'));
 const bands={ok:{n:70},weak:{n:20},bad:{n:10},na:{n:0}};
 const dash={ok:true,available:true,window:{date_from:'2026-09-01',date_to:'2026-09-05'},zones:Object.fromEntries(['zone1','zone2a','zone2b'].map(k=>[k,{avg_pct:78,bands}])),reg55:{available:true,avg_pct:82,band_share:{compliant_min:{n:70},compliant_measures:{n:20},noncompliant:{n:10},unscored:{n:0}}},trends:Array.from({length:5},(_,i)=>({date:`2026-09-0${i+1}`,zone1_avg:70+i,zone2a_avg:78+i,zone2b_avg:80-i,reg55_avg:82}))};
 const family=id=>({id,cases:10,pct:10,tiles:[{id:'unused',title_ru:id==='lab'?'Анализы не учтены':'Взаимодействия',codes:[id==='lab'?'B_lab_unused_in_dx':'C_ddi'],cases:10,pct:10,denominator:id==='lab'?'cases_with_lab':'total_cases',denominator_n:100}],by_code:[{code:id==='lab'?'B_lab_unused_in_dx':'C_ddi',title_ru:id==='lab'?'Готовый анализ не учтён в диагнозе':'Лекарственное взаимодействие',cases:10,pct:10}],by_specialty:[{specialty:'Специальность A',cases:10,pct:10}],by_doctor:[{doctor:'Демонстрационный врач',cases:1,pct:1}]});
 await page.route('**/*',async route=>{
  const u=new URL(route.request().url());
  if(u.hostname!=='mo-audit.local')return route.abort();
  const p=u.pathname;
  if(p.startsWith('/api/')){
   requests.push(p+u.search);
   let data={ok:true,items:[],rows:[],facets:{},data_through:'2026-09-05'};
   if(p.endsWith('/capabilities'))data={ok:true,pages:Object.fromEntries(['yesterday','overview','queue','documents','doctors','medications','labs','reports','kp-sync','rceth-sync','settings'].map(p=>[p,true])),actions:{}};
   if(p.endsWith('/drugs-labs-kpis'))data={ok:true,families:{lab:family('lab'),drug:family('drug')},denominators:{total_cases:100,lab_coverage_available:false},strips:{drug:{pct:10},lab:{pct:10}}};
   if(p.endsWith('/score-dashboard'))data=dash;
   if(p.endsWith('/daily-report'))data={ok:true,date:'2026-09-05',data_through:'2026-09-05',attention:{n_evaluated:100},data_completeness:{},actions:[],summary:{n:100},axes:{}};
   if(p.endsWith('/month-report'))data={...data,period:{date_from:'2026-09-01'},kpi:{source_records:100,evaluated:100},data_through:'2026-09-05'};
   return route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(data)});
  }
  let file=p==='/methodist/mo'?path.join(root,'frontend/web/methodist/mis-kz-quality.html'):path.join(root,'frontend/web/shared',p);
  if(fs.existsSync(file))return route.fulfill({path:file});
  return route.fulfill({status:404,body:''});
 });
 const results=[];
 for(const name of ['yesterday','overview','queue','documents','doctors','medications','labs','reports','kp-sync','rceth-sync','settings']){
  await page.goto('http://mo-audit.local/methodist/mo?page='+name+'&period=month');
  await page.waitForTimeout(350);
  results.push({page:name,visible:await page.locator('#page-'+name).isVisible(),charts:await page.locator('#page-'+name+' canvas').count(),overflow:await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)});
  if(['yesterday','medications','labs'].includes(name))await page.screenshot({path:'/private/tmp/mo-audit-artifacts/'+name+'-desktop.png',fullPage:true});
 }
 await page.goto('http://mo-audit.local/methodist/mo?page=labs&period=month');
 await page.waitForTimeout(200);
 results.push({labSubtitle:await page.locator('#labs-coverage').innerText(),tileMeta:await page.locator('#labs-kpis .kpi-meta').innerText(),tableRowKeyboardFocusable:await page.locator('#labs-codes tbody tr').evaluate(el=>el.tabIndex>=0)});
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:'/private/tmp/mo-audit-artifacts/labs-mobile.png',fullPage:true});
 results.push({mobileDocumentWidth:await page.evaluate(()=>document.documentElement.scrollWidth),mobileViewport:390});
 fs.writeFileSync('/private/tmp/mo-audit-artifacts/ui-results.json',JSON.stringify({results,errors,requests},null,2));
 console.log(JSON.stringify({results,errors},null,2));
 await browser.close();
})();
