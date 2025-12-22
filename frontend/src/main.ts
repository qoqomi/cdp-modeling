import { createApp } from "vue";
import { createPinia } from "pinia";
import { VueQueryPlugin } from "@tanstack/vue-query";
import { MotionPlugin } from "@vueuse/motion";
import App from "./App.vue";
import router from "./router";
import "./index.css";
import "./styles/globals.css";

const app = createApp(App);
const pinia = createPinia();

app.use(pinia);
app.use(router);
app.use(VueQueryPlugin);
app.use(MotionPlugin);
app.mount("#app");
