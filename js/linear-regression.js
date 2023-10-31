import {render} from '@tensorflow/tfjs-vis'
 
    window.onload = function () {
        const x_data = [1,3,5,7,9]
        const y_data = [2,4,6,8,10]

        const series = ['X轴', 'Y轴']
        const data = {values: [x_data, y_data], series}

        const surface = {
            name: '散点图',
            tab: '散点图表',
            data
        }

        render.scatterplot(surface)
    }