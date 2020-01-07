import Vue from 'vue'
import VueRouter from 'vue-router'
import Train from '../views/Train.vue'
import Predict from '../views/Predict.vue'

Vue.use(VueRouter)

const routes = [{
    path: '/',
    name: 'home',
    redirect: '/train'
  },{
    path: '/train',
    name: 'train',
    component: Train
  },{
    path: '/predict',
    name: 'predict',
    component: Predict
  }
]

const router = new VueRouter({
  routes
})

export default router
