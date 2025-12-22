import { createRouter, createWebHistory } from "vue-router";
import type { RouteRecordRaw } from "vue-router";

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    redirect: "/dashboard/proof/test",
  },
  {
    path: "/dashboard",
    name: "dashboard",
    component: () => import("../layouts/DashboardLayout.vue"),
    redirect: "/dashboard/proof",
    children: [
      {
        path: "proof",
        name: "proof",
        component: () => import("../views/proof/ProofListPage.vue"),
        meta: { title: "CDP/DJSI 증빙분석" },
      },
      {
        path: "proof/:reportId",
        name: "proof-analysis",
        component: () => import("../views/proof/ProofAnalysisPage.vue"),
        meta: { title: "CDP/DJSI 증빙분석" },
      },
    ],
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
