using Flux
using Lathe.preprocess: TrainTestSplit
using DataFrames
using CSV
using StatsBase
using Dates
using TimeZones

# PRIPREMA PODATAKA
df = DataFrame(CSV.File("weatherHistory.csv"))
select!(df, Not(:Summary))
select!(df, Not(:PrecipType))
select!(df, Not(:DailySummary))
select!(df, Not(:LoudCover))


function razlika_u_sekundama(do_datuma)
    # Parsiraj zadati datum
 
    datum = ZonedDateTime(String(do_datuma), "yyyy-mm-dd HH:MM:SS.SSS z")

    referenca=df.FormattedDate[1]
    #referenca=string(referenca[1:27], ":", referenca[28:end])
    referenca = ZonedDateTime(String(referenca), "yyyy-mm-dd HH:MM:SS.SSS z")

    # Izračunaj razliku u sekundama
    datum = DateTime(datum)
    referenca = DateTime(referenca)
    rez = Dates.datetime2epochms(datum) - Dates.datetime2epochms(referenca)
    rez=rez/3600000    

    return rez
end

#formatiranje datuma
    niz=[]

for i in 1:length(df.FormattedDate)
    
     push!(niz,razlika_u_sekundama("$(df.FormattedDate[i])"))
    
end

df[!,:Vreme]=niz

#izbacivanje prve kolone jer je vise ne koristimo
select!(df, Not(:FormattedDate))


#Normalizacija
mean1=mean(df.Vreme)
S=std(df.Vreme)
@. df.Vreme = ((df.Vreme -mean1) / S)


data_train_x, data_Test_x = TrainTestSplit(df, 0.75)

data_train_y = select(data_train_x,:Temperature)
select!(data_train_x, Not(:Temperature))

data_Test_y = data_Test_x.Temperature
select!(data_Test_x, Not(:Temperature))


data_train_x =  convert(Array{Float32},data_train_x)'
data_train_y = convert(Array{Float32},data_train_y)'
data_Test_x = convert(Array{Float32},data_Test_x)'
data_Test_y = convert(Array{Float32},data_Test_y)'

# Definisanje modela

model = Dense(7=>1)


loss(x, y) = Flux.mse(model(x), y) 
params_model = Flux.params(model) 
opt = Adam(0.02) 
dataset = [(data_train_x, data_train_y)] 
#display(describe(dataset))

tt = 20000 
for i in 1:tt
    Flux.train!(loss, params_model, dataset, opt) 
end

#trening greska
y_model = model(data_train_x) 
errors_train = data_train_y - y_model
MSE = mean(abs.(errors_train .* errors_train)) 

#test greska
y_test=model(data_Test_x)
errors_Test=data_Test_y-y_test
MSE1 = mean(abs.(errors_Test .* errors_Test))

MAE = mean(abs.(errors_Test))
RMSE=sqrt(MSE1)


data_Test_y .= map(x -> x == 0 ? 1 : x, data_Test_y)

MAPE=mean(abs.(model(data_Test_x).-data_Test_y)./data_Test_y)*100

println("GRESKA PRILIKOM TRENIRANJA JE: " ,MSE)
println("GRESKA PRILIKOM TESTIRANJA JE: " ,MSE1)
println("")
println("MAE: ",MAE)
println("RMSE: ",RMSE)
println("MAPE: ",MAPE,"%")