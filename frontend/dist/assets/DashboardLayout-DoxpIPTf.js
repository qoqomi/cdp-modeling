import{k as n,d as C,l as _,r as M,c as d,a as s,m as o,n as h,p as e,f as y,g as m,F as S,h as z,q as p,t as u,s as k,u as w,v as D,b as l,x as I,y as B,_ as F}from"./index-C4UDD_tn.js";import{C as L}from"./chart-column-Bcy9-aHq.js";import{F as V}from"./file-search-CyA5plmm.js";/**
 * @license lucide-vue-next v0.460.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const q=n("ChevronLeftIcon",[["path",{d:"m15 18-6-6 6-6",key:"1wnfg3"}]]);/**
 * @license lucide-vue-next v0.460.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const E=n("EarthIcon",[["path",{d:"M21.54 15H17a2 2 0 0 0-2 2v4.54",key:"1djwo0"}],["path",{d:"M7 3.34V5a3 3 0 0 0 3 3a2 2 0 0 1 2 2c0 1.1.9 2 2 2a2 2 0 0 0 2-2c0-1.1.9-2 2-2h3.17",key:"1tzkfa"}],["path",{d:"M11 21.95V18a2 2 0 0 0-2-2a2 2 0 0 1-2-2v-1a2 2 0 0 0-2-2H2.05",key:"14pb5j"}],["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}]]);/**
 * @license lucide-vue-next v0.460.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const H=n("FileCheckIcon",[["path",{d:"M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",key:"1rqfz7"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"m9 15 2 2 4-4",key:"1grp1n"}]]);/**
 * @license lucide-vue-next v0.460.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const T=n("MessagesSquareIcon",[["path",{d:"M14 9a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2z",key:"p1xzt8"}],["path",{d:"M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1",key:"1cx29u"}]]);/**
 * @license lucide-vue-next v0.460.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const $=n("MoonIcon",[["path",{d:"M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z",key:"a7tn18"}]]);/**
 * @license lucide-vue-next v0.460.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const j=n("SunIcon",[["circle",{cx:"12",cy:"12",r:"4",key:"4exip2"}],["path",{d:"M12 2v2",key:"tus03m"}],["path",{d:"M12 20v2",key:"1lh1kg"}],["path",{d:"m4.93 4.93 1.41 1.41",key:"149t6j"}],["path",{d:"m17.66 17.66 1.41 1.41",key:"ptbguv"}],["path",{d:"M2 12h2",key:"1t8f8n"}],["path",{d:"M20 12h2",key:"1q8mjw"}],["path",{d:"m6.34 17.66-1.41 1.41",key:"1m8zz5"}],["path",{d:"m19.07 4.93-1.41 1.41",key:"1shlcs"}]]),N={class:"relative z-10 h-screen flex flex-col",style:{"box-shadow":"none"}},R={class:"flex-1 flex overflow-hidden p-4 gap-4",style:{"box-shadow":"none"}},A={class:"nav-menu"},G=["title"],Z={class:"flex-1 flex overflow-visible",style:{"box-shadow":"none"}},P=C({__name:"DashboardLayout",setup(U){const c=D(),v=w(),a=_(),r=M(!0),b=[{id:"risk",label:"TCFD 분석 HUB",icon:L,badge:null},{id:"tnfd",label:"TNFD 에이전트",icon:H,badge:null},{id:"proof",label:"외부 ESG 평가 대응",icon:V,badge:null},{id:"copilot",label:"ESG AI 어시스턴트",icon:T,badge:null}],g=()=>{v.push("/")};return(X,i)=>{const f=k("router-link"),x=k("router-view");return l(),d("div",N,[s("div",R,[s("aside",{class:h(["sidebar flex-shrink-0 flex flex-col",{collapsed:r.value,dark:e(a).isDark}]),style:o({backgroundColor:e(a).isDark?"rgba(17, 17, 17, 0.98)":"rgba(255, 255, 255, 0.98)",borderColor:e(a).theme.border3d.dark,boxShadow:e(a).theme.elevation.card})},[s("div",{class:"sidebar-header",style:o({borderColor:e(a).theme.cardBorder})},[s("div",{class:"logo-wrapper",onClick:g},[s("div",{class:"logo-icon",style:o({backgroundColor:`${e(a).theme.primary}15`})},[m(e(E),{size:22,style:o({color:e(a).theme.primary})},null,8,["style"])],4),r.value?y("",!0):(l(),d("span",{key:0,class:"logo-text",style:o({color:e(a).theme.textPrimary,border:"none"})}," CLIMA-X ",4))]),s("button",{class:"toggle-btn",style:o({backgroundColor:e(a).theme.primary}),onClick:i[0]||(i[0]=t=>r.value=!r.value)},[m(e(q),{size:16,class:h([{"rotate-180":r.value},"toggle-icon"])},null,8,["class"])],4)],4),s("nav",A,[(l(),d(S,null,z(b,t=>m(f,{key:t.id,to:`/dashboard/${t.id}`,class:h(["nav-item",{active:e(c).name===t.id}]),style:o(e(c).name===t.id?{backgroundColor:`${e(a).theme.primary}20`,borderRadius:"14px"}:{})},{default:I(()=>[s("div",{class:h(["nav-icon-standalone",{"active-icon":e(c).name===t.id}]),style:o({color:e(c).name===t.id?e(a).theme.primary:e(a).theme.textSecondary})},[(l(),p(B(t.icon),{size:20}))],6),r.value?y("",!0):(l(),d("span",{key:0,class:"nav-label",style:o({color:e(c).name===t.id?e(a).theme.primary:e(a).theme.textSecondary})},u(t.label),5))]),_:2},1032,["to","class","style"])),64))]),s("div",{class:"sidebar-footer",style:o({borderColor:e(a).theme.cardBorder})},[s("button",{class:"theme-toggle-btn",style:o({backgroundColor:e(a).isDark?"rgba(255, 255, 255, 0.08)":"rgba(0, 0, 0, 0.03)"}),title:e(a).isDark?"밝은 테마로 전환":"어두운 테마로 전환",onClick:i[1]||(i[1]=t=>e(a).toggleTheme())},[e(a).isDark?(l(),p(e(j),{key:0,size:18,style:{color:"#fbbf24"}})):(l(),p(e($),{key:1,size:18,style:o({color:e(a).theme.textSecondary})},null,8,["style"])),r.value?y("",!0):(l(),d("span",{key:2,class:"theme-label",style:o({color:e(a).theme.textSecondary})},u(e(a).isDark?"라이트 모드":"다크 모드"),5))],12,G)],4)],6),s("main",Z,[m(x)])])])}}}),Q=F(P,[["__scopeId","data-v-b57d2163"]]);export{Q as default};
