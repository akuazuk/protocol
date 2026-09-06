const {chromium}=require('playwright');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
let browser;
(async()=>{
 const root=process.cwd(); browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.addInitScript(()=>localStorage.setItem('protocol_methodist_token','synthetic-local-only'));
 let scores;
 await page.route('**/*',async route=>{
  const u=new URL(route.request().url());
  if(u.hostname!=='mo-ui.local')return route.abort();
  if(u.pathname.startsWith('/api/')){
   let data={ok:true,items:[],rows:[],facets:{}};
   if(u.pathname.endsWith('/capabilities'))data={ok:true,pages:{documents:true,labs:true},actions:{}};
   if(u.pathname.endsWith('/drugs-labs-kpis'))data={families:{lab:{by_code:[{code:'B_lab_unused_in_dx',title_ru:'Тестовое замечание',cases:1,pct:1}],by_doctor:[{doctor:'Тестовый врач',cases:1,pct:1}]}},denominators:{total_cases:100}};
   if(u.pathname.endsWith('/cases/synthetic'))data={record:{case_id:'synthetic',parse_ok:1},family_scores:scores};
   return route.fulfill({contentType:'application/json',body:JSON.stringify(data)});
  }
  const f=u.pathname==='/methodist/mo'?path.join(root,'frontend/web/methodist/mis-kz-quality.html'):path.join(root,'frontend/web/shared',u.pathname);
  return fs.existsSync(f)?route.fulfill({path:f}):route.fulfill({status:404,body:''});
 });
 await page.goto('http://mo-ui.local/methodist/mo?page=documents');
 await page.evaluate(()=>{const b=document.createElement('button');b.dataset.openCase='synthetic';b.id='synthetic-open';b.textContent='Тестовый осмотр';document.getElementById('yesterday-findings-list').append(b);});
 const fixtures=[
  [undefined,['Не оценено','Не оценено'],'Недостаточно данных'],
  [{drug_score:null,lab_score:null},['Не оценено','Не оценено'],'Недостаточно данных'],
  [{drug_score:0,lab_score:60,drug:{status:'completed'},lab:{status:'partial'}},['0 / 100','60 / 100'],'Частичная проверка'],
  [{drug_score:100,lab_score:100},['100 / 100','100 / 100'],'Полнота проверки не подтверждена']
 ];
 for(const [fixture,values,note]of fixtures){
  scores=fixture;await page.locator('#synthetic-open').evaluate(el=>el.click());
  await page.locator('.family-score-chip').first().waitFor();
  assert.deepEqual(await page.locator('.family-score-chip .kpi-value').allTextContents(),values);
  const confidence=page.locator('.drawer-grid .kpi').filter({hasText:'Уверенность расчёта'});
  assert((await confidence.textContent()).includes('Не предоставлена'));
  assert(!(await confidence.textContent()).includes('90'));
  assert((await page.locator('.family-score-row').innerText()).includes(note));
  await page.locator('#drawer-close').click();
 }
 scores={drug_score:null,lab_score:null};await page.locator('#synthetic-open').evaluate(el=>el.click());await page.locator('.family-score-chip').first().waitFor();
 await page.locator('.family-score-row').screenshot({path:'/private/tmp/mo-family-ui-desktop.png'});
 await page.setViewportSize({width:390,height:844});
 await page.locator('.family-score-row').screenshot({path:'/private/tmp/mo-family-ui-mobile.png'});
 const sizes=await page.locator('.family-score-row').evaluate(el=>({width:el.clientWidth,scroll:el.scrollWidth}));
 assert(sizes.scroll<=sizes.width,'family row overflow');assert.deepEqual(errors,[]);
 await page.locator('#drawer-close').click();
 await page.goto('http://mo-ui.local/methodist/mo?page=labs');
 const drill=page.locator('#labs-codes button');await drill.waitFor();
 assert((await page.locator('#labs-doctors').textContent()).includes('Доля всех МО периода'));
 await drill.focus();await page.keyboard.press('Enter');
 await page.waitForURL(u=>u.searchParams.get('page')==='documents' && u.searchParams.get('finding_codes')==='B_lab_unused_in_dx');
 console.log(JSON.stringify({fixtures:fixtures.length,errors,mobile:sizes,keyboardDrill:true}));await browser.close();
})().catch(async e=>{console.error(e);if(browser)await browser.close();process.exitCode=1});
