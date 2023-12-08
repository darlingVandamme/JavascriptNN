
const counters ={}

function count(label){
    if(counters[label]) {
        counters[label]++
    } else {
        counters[label]=1
    }
}

function countlog(){
    console.log("counters ",counters)
}


export {countlog,count}


