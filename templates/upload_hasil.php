<!DOCTYPE html>
<html>
<head>
	<title> Pengenalan Alfabet Bahasa Isyarat Tangan </title>
</head>
<body style="background: linear-gradient(0deg, #dedeff, white 70%) no-repeat">
	<div class="container" align="center">
		<p style="font-size: 200%; font-weight: bold"> SIGN LANGUAGE ALPHABET RECOGNITION </p>
		<div id="isi" style="width: 55%; height: 500px; background-color: #fdfdff; border: 3px inset blue;">
			<p style='font-size: 25px; font-weight: bold'> Hasil Akhir </p>
			<form method="post" action="">
				<video width="240" height="240" controls style="float: left; padding-left: 4%">
					<source src="{{ name }}" type="video/mp4">
				</video>
				<p style="float: left; font-size: 70px; padding-left: 3%"> &rarr; </p>
				<img src="../static/{{ file }}-0.jpg" width="80px" height="80px">
				<img src="../static/{{ file }}-1.jpg" width="80px" height="80px">
				<img src="../static/{{ file }}-2.jpg" width="80px" height="80px">
				<img src="../static/{{ file }}-3.jpg" width="80px" height="80px">
				<img src="../static/{{ file }}-4.jpg" width="80px" height="80px">
				<img src="../static/{{ file }}-5.jpg" width="80px" height="80px">
				<img src="../static/{{ file }}-6.jpg" width="80px" height="80px">
				<img src="../static/{{ file }}-7.jpg" width="80px" height="80px">
				<img src="../static/{{ file }}-8.jpg" width="80px" height="80px">
				<img src="../static/{{ file }}-9.jpg" width="80px" height="80px">
				<p style='font-size: 25px;'> Hasil Klasifikasi : </p>
				<div id="hasil"  style='font-size: 30px; font-weight: bold'> {{ hasil }} </div>
			</form>
		</div>
	</div>
</body>
</html>