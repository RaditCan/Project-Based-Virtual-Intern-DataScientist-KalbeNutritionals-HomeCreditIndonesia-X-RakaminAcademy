select avg(age)
from tbcustomer
where "Marital Status" = 'Married';

select avg(age)
from tbcustomer
where "Marital Status" = 'Single';

select avg(age)
from tbcustomer
where gender =0;

select avg(age)
from tbcustomer
where gender =1;


SELECT tbstore.storename  , sum(qty) as Totalqty 
from tbtransaction
inner join tbstore ON tbtransaction.storeid  = tbstore.storeid 
group by tbstore.storename 
ORDER BY totalqty DESC;

SELECT tbproduct."Product Name"  , sum(totalamount) as JumlahTotalAmount 
from tbtransaction
inner join tbproduct ON tbtransaction.productid  = tbproduct.productid 
group by tbproduct."Product Name" 
ORDER BY JumlahTotalAmount DESC;


