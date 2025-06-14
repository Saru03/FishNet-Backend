
<!DOCTYPE html>

<html lang="en">

<head>


<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<meta name="MobileOptimized" content="width">
<meta name="HandheldFriendly" content="true">


<title>NASA Ocean Color</title>


<!-- SCRIPTS -->
<script src='https://oceancolor.gsfc.nasa.gov/js/jquery-3.7.1.min.js'></script>
<script src='https://oceancolor.gsfc.nasa.gov/js/random_image.js'></script>
<!--<script src='https://oceancolor.gsfc.nasa.gov/includes/nav_data.js'></script>-->

<script type="text/javascript" src="/globalassets/static/js/js_functions.js"></script>

<script src='https://oceancolor.gsfc.nasa.gov/includes/nav_data.js'></script>
<!-- Google Tag Manager -->
<script>
(
	function(w,d,s,l,i) {
		w[l]=w[l]||[];
		w[l].push(
			{
				"gtm.start": new Date().getTime(),
				event:"gtm.js"
			}
		);
		var f=d.getElementsByTagName(s)[0];
		var j=d.createElement(s);
		var dl=l!="dataLayer"?"&l="+l:"";
		j.async=true;
		j.src="https://www.googletagmanager.com/gtm.js?id="+i+dl;f.parentNode.insertBefore(j,f);
	}
)
(window,document,"script","dataLayer","GTM-WNP7MLF");
</script>
<!-- End Google Tag Manager -->


<!-- We participate in the US government's analytics program. See the data at analytics.usa.gov. -->
<script async type="text/javascript" src="https://dap.digitalgov.gov/Universal-Federated-Analytics-Min.js?agency=NASA&subagency=GSFC" id="_fed_an_ua_tag"></script>


<!-- FONTS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-Fy6S3B9q64WdZWQUiU+q4/2Lc9npb8tCaSX9FK7E8HnRr0Jz8D6OP9dO5Vg3Q9ct" crossorigin="anonymous"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer" />



<!-- RESPONSIVE NAV FILES -->
<link href="https://oceancolor.gsfc.nasa.gov/css/nav-files/bootstrap.css" rel="stylesheet">
<link href="https://oceancolor.gsfc.nasa.gov/css/nav-files/smartmenus-1.2.1/addons/bootstrap/jquery.smartmenus.bootstrap.css" rel="stylesheet">
<!-- END RESPONSIVE NAV FILES -->


<!-- SmartMenus jQuery plugin -->
<script type="text/javascript" src="https://oceancolor.gsfc.nasa.gov/js/nav-files/smartmenus-1.2.1/jquery.smartmenus.js"></script>
<script type="text/javascript" src="https://oceancolor.gsfc.nasa.gov/js/nav-files/smartmenus-1.2.1/addons/bootstrap/jquery.smartmenus.bootstrap.js"></script>

<!-- EARTH DATA -->
<link href="https://oceancolor.gsfc.nasa.gov/css/application.css" rel="stylesheet" />
<script src="https://cdn.earthdata.nasa.gov/eui/1.1.8/js/eui.js"></script>


<!-- CUSTOM STYLES -->
<link href="https://oceancolor.gsfc.nasa.gov/css/styles.css" rel="stylesheet">
<link href="https://oceancolor.gsfc.nasa.gov/css/navigation.css" rel="stylesheet">

<link href="/globalassets/static/css/w3.css" rel="stylesheet" />
<link href="/globalassets/static/css/oceandata.css" rel="stylesheet" />


</head>

<body>


<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-WNP7MLF" height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->


<div class="full-width-container">
<!--<script src="https://oceancolor.gsfc.nasa.gov/js/tophat.js" id="earthdata-tophat-script" data-show-fbm="false" data-show-status="true" data-status-polling-interval=5 data-current-daac="OB.DAAC" data-show-outage-banner="true" data-use-fontawesome="true"></script>-->
<script nonce="668206489" src="https://cdn.earthdata.nasa.gov/tophat2/tophat2.js" id="earthdata-tophat-script" data-show-fbm="false" data-show-status="true" data-status-polling-interval=5 data-current-daac="OB.DAAC" data-show-outage-banner="true" data-use-fontawesome="false"></script>
</div>


<div class="full-width-container">

	<div class="oc-header-grid">

		<!--<div class="oc-header-item"><a href="https://oceancolor.gsfc.nasa.gov/"><img class="oc-obdaac-logo-image" src="https://oceancolor.gsfc.nasa.gov/images/ob-logo-svg-2.svg" alt="OB.DAAC Logo" /></a></div>-->
		<div class="oc-header-item"><a href="https://oceancolor.gsfc.nasa.gov/">

			<svg version="1.2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 513 110" width="513" height="110" class="oc-obdaac-logo-image" role="img" aria-label="OB.DAAC Logo">
				<desc>A sphere with three ocean waves in differing shades of blue.  Next to the sphere, there is the bolded text "Ocean Color".  Under that, there are two acronyms, separated by a vertical pipe: OB.DAAC (Ocean Biology Distributed Active Archive Center) and OBPG (Ocean Biology Processing Group).</desc>
				<defs>
					<linearGradient id="g1" x1="163.5" y1="31.5" x2="253.5" y2="31.5" gradientUnits="userSpaceOnUse">
						<stop offset="0" stop-color="#ff0000"/>
						<stop offset="1" stop-color="#ffa500"/>
					</linearGradient>
					<linearGradient id="g2" x1="423.5" y1="281.5" x2="553.5" y2="411.5" gradientUnits="userSpaceOnUse">
						<stop offset="0" stop-color="#d150ff"/>
						<stop offset="1" stop-color="#0000ff"/>
					</linearGradient>
					<clipPath clipPathUnits="userSpaceOnUse" id="cp1">
						<path d="m2.65 1.86h105.23v54.4h-105.23z"/>
					</clipPath>
					<clipPath clipPathUnits="userSpaceOnUse" id="cp2">
						<path d="m44.84 70.18h62.6v38.57h-62.6z"/>
					</clipPath>
					<clipPath clipPathUnits="userSpaceOnUse" id="cp3">
						<path d="m3.44 35.79h106v62.44h-106z"/>
					</clipPath>
				</defs>
				<style>
					tspan { white-space:pre } 
					.t0 { font-size: 20px;fill: #000000;font-family: "' Open  Sans'" }
					.t1 { font-size: 20px;fill: #ff0000;font-family: "' Impact'" }
					.s2 { fill: url(#g1) }
					.s3 { fill: url(#g2) }
					.s4 { fill: #000000;stroke: #ff0000 }
					.s5 { opacity: .5;fill: #008000;stroke: #000000;stroke-linecap: square;stroke-width: 4;stroke-dasharray: 16 }
					.s6 { fill: #231f20 }
					.s7 { fill: #ffffff }
					.s8 { fill: #000000 }
					.s9 { fill: #52ace4 }
					.s10 { fill: #25295e }
					.s11 { fill: #2c3b92 }
				</style>
				<g id="Layer">
					<g id="Layer">
						<text id="Lorem " style="transform: matrix(1,0,0,1,433.5,219.08)" >
							<tspan x="0" y="0" class="t0">Lorem 
			</tspan>
						</text>
						<text id="ipsum" style="transform: matrix(1,0,0,1,555.75,219.08)" >
							<tspan x="0" y="0" class="t0">ipsum
			</tspan>
						</text>
						<text id="dolor " style="transform: matrix(1,0,0,1,543.5,245.3)" >
							<tspan x="0" y="0" class="t0">dolor 
			</tspan>
						</text>
						<text id="sit" style="transform: matrix(1,0,0,1,653.53,245.3)" >
							<tspan x="0" y="0" class="t1">sit
			</tspan>
						</text>
						<text id=" amet" style="transform: matrix(1,0,0,1,704.52,245.3)" >
							<tspan x="0" y="0" class="t0"> amet
			</tspan>
						</text>
					</g>
				</g>
				<g id="Layer">
					<path id="Layer" class="s2" d="m163.5 31.5h90v50h-90z"/>
				</g>
				<g id="Layer">
					<path id="Layer" class="s3" d="m488.5 411.5c-35.9 0-65-29.1-65-65 0-35.9 29.1-65 65-65 35.9 0 65 29.1 65 65 0 35.9-29.1 65-65 65z"/>
				</g>
				<g id="Layer">
					<path id="Layer" class="s4" d="m83.5 81.5l40 20-10 40 20-20 40 100-60-50"/>
				</g>
				<g id="Layer">
					<path id="Layer" class="s5" d="m532 486l200 100-30 100z"/>
				</g>
				<g id="Layer">
					<path id="Layer" fill-rule="evenodd" class="s6" d="m387.9 312.4c-1.2 0-3.1-3-3.1-3-0.3 4-4.8 9.1-8.1 9.1-3.3 0-1.4-0.7-4.4-0.7-2.9 0-1.5 3.7-1.5 3.7-5.9-4-0.7-8.4-0.7-8.4-3.8 0-2.8-1.5-6.1-1.9-3.3-0.5-5.9 2.6-5.9 2.6 1-7.6 6.1-6.8 6.1-6.8 0 0 0.1-1.5 2.2-1.3 3.9 0 4.9 2.8 8.7 2.8 3.8 0 4.2-4.1 3.5-6.9-0.6-2.9-4-6.1-16.1-6.1-10 0-16.3-11.7-18.9-17.9-1.1 1.9-5.9 6.6-5.9 10.2 0 4.1 0.9 5.6 2.3 6.5 1.4 0.8 0.7 3.3-1.1 4-1.7 0.7-10.7 9.3-10.7 11.6 0 0 2.9-0.3 4.9-1.5 2-1.2 5.3 2.3 5.3 5.1 0 2.8-4.3 4.4-4.3 4.4 0 0 1.7-3.1-1.1-3.8-2.7-0.7-8.6-0.2-9.4-1.2-0.9-1-3-0.7-3.8 0-0.7 0.6-5.9 1.1-8.1 0.9-2.2-0.1-5.7 0.4-5.4 3.3 0 0-1.9-3.2 0.2-6.1 1.7-2.2 2.8-2.6 3.2-2.6-0.8 0-4.1-0.1-4.6-1-0.7-1-1.5-0.9-2.5-0.6-1.1 0.2-4.4 3.8-4.4 6.6 0 0-1.5-8.3 6.1-10.1 0 0 0.5-1 2.8-1 2.3 0 5.3 3.6 5.3 3.6 0 0 0.4-2.4 1.8-2.4 1.5 0 2.1 4.5 6.2 4.5 4 0 8.5-5.5 9.8-7.2 1.2-1.8 1.8-2.8 0.6-5.2-1.3-2.4-5.1-6.9-4.1-9.9 1-2.9 2-8.3 1.9-10-0.8 0.4-4.5 2-5.8 3.8-1.4 2-6 2.3-6 2.3-1.4 2.6-8.5 2.4-8.5 2.4-2.2 1.6-9.3 0.7-11.3 0.4-0.7 0.6-1.6 1.3-2.8 2-3 1.7-12.4 11.1-14.4 12.3-2 1.2-2.6 3.8-2.6 3.8 0 0-1.3-1-2.6 0.1-1.2 1.2 1.9 2.8 3.2 1.8 1.2-1.1 5.5-1.2 6.1 2.5 0.6 3.7-2.6 4.8-2.6 4.8 0 0 0.3-0.5 0.3-2.1 0-1.7-1.2 0.5-4.6 0.5-3.4 0-2.6-3-7.4-3-4.7 0-6.9 4.7-11.7 4.7-4.7 0-5.6-2.4-5.6-2.4-3 1.8-1.2 6.8-1.2 6.8 0 0-3.6-3-2.5-6.9 1.1-3.9 5.5-3.3 5.5-3.3 0 0-3.7-1.6-6.3-1.6-2.6-0.1-2.9-1.7-2.9-1.7-4.6 4.2-1.9 7.2-1.9 7.2 0 0-2.5-1-2.5-5.1 0-3.2 2.2-4.7 3.2-5.2-0.1 0-0.1-0.3 0.4-1 1-1.6 5-1.1 5.9 0 0.9 1 3.6 2.1 3.6 2.1 0 0 0 0 0.1-0.9 0.2-1 1.9-2.4 2.7-1.3 0.8 1.1 2.9 2.3 4.2 3.1 1.3 0.8 5.6 0 5 0-0.6-0.1-0.8-1.1-0.8-1.1 0 0 2.1 0.6 3.6-0.5 1.4-1 8.8-9.1 10.8-13.7 0.8-1.8 1.8-3.1 2.7-4.1-0.7-0.3-2.1-1-4.8-2.8q-1.3-0.9-2.2-1.9c-1.1-0.1-1 3.6-1 3.6 0 0-1.4-0.7-6.4-0.7-4.9 0-13.2-4.8-15.8-7.2-2.6-2.4-5.4-2.3-8.2-2-2.8 0.2-3.2 0.5-2.8 1.3 0.5 0.8 1 1.1 4.8 0.5 3.7-0.7 4 4.2 3.3 6.6-0.8 2.4-5.7 2-5.7 2 0 0 1.3-0.3 2.7-1.2 1.4-0.9 2-3.1-2.3-3.4-3.7-0.3-5.2-2.2-5.5-2.6q-0.1 0-0.4 0c-2.8-0.1-7.1 1.7-8 2.5-0.9 0.8-2.5 1.3-3.6 0.7-1.2-0.6-3.2 3-3.2 3-1.3-3.3 2.1-4.7 1.9-5.7-0.2-1 0.9-2.3 0.9-2.3 0 0-0.5 0-2.8 0-2.2 0-3.8-1.2-3.8-1.2-5.9 1.9-3 4.9-3 4.9 0 0-2.5-0.1-2.5-3 0-2.8 4.7-4 4.7-4 0 0 1.3-2.1 3.5-2.1 2.2 0 3.2 0.2 3.2 0.2 0 0 1.2-2.6 3.6-1.6 2.3 1.1 5.6 2.3 7.6 2.3 2.1 0 6.4-2.2 8-3.5 1.6-1.3 4.1-0.6 5.8 1.4 1.7 2.1 5.4 2.1 11.3 2.7 3.1 0.3 5 1.4 6.1 2.3-0.6-0.6-1.2-1.3-1.7-1.8-1.2-1-3.5-3.8-0.2-10.1 2.5-4.7 4.4-5.1 5.3-5 0.3-1.5 0.5-2.4 0.7-2.6 1.7-7.1 8.1-12 11.2-13.6 3.4-1.8 3.9-2.5 4.3-4.4 0.3-2-2.1-1-4.7-0.6l-0.2-0.4c-3 1.8-7.3 4.6-8.1 5.7-1.2 1.6-3.6 5.1-6.5 4.1 0 0 1.7-0.5 1.5-1.6-0.3-1 0.1-2.6 0.1-2.6 0 0-1.1 1-1.1 1.8 0 0.8-1.6 1.3-1.6 1.3 0 0 1.1-2.5 0.5-3.6-0.6-1-0.7-3.7 0.2-5.4 0 0 0.4 2.8 2.2 3 0 0-1.3-3.7 1.7-5.7 0 0-0.8 4.8 1.7 5 2.5 0.2 3.9-1.2 5.4-2 1.6-0.9 2.8-2 2.7-4-0.1-1.9-1.2-8.9-5.5-9-4.3-0.2-4.6 1.3-3.6 4 0 0-2.7 0.1-2.6-4.5 0 0-2.5-1-2.5 1.4 0 2.4 0.1 3.1 0.9 3.5 0 0-2.1-0.7-1.9-5.2 0.2-4.6 2.5-6.4 5.7-6.7 3.2-0.4 6.2 3.1 9.1 3.1 2.8 0 2.9-0.7 2.8-2-0.1-1.2 2-1.8 3.4-0.2 1.4 1.6 1.4 1.6 1.4 1.6 0 0 1.2-0.7-0.2-2.5 0 0 3.2-0.1 4.3 2.1 0 0-0.3-2.2 0.8-6 1.1-3.9 2.1-6.4 2.1-9.1 0 0 0.5 7 1.1 9.2 0.6 2.1 1.6 7 1.3 9.8 0 0 1.1-1.8 2.9-2.2-0.8-12.4-5-28.4-18.4-44.5 0 0 9.9 5.9 12.8 10.7 2.9 4.8 6.1 9.9 9.9 9.9 3.7 0 1.3-14.5 1.3-14.5 0 0 4.6 8.6 4.6 12.9 0 4.2 7.4 8 8.2-3.8 0 0 1.3 7 1.2 9.7-0.1 2.8 0.8 5.3 2.4 5.5 1.6 0.1 2.8-5.7 2.8-5.7 0 0 0.6 6.5 0.3 8.6-0.3 2.1 2.9 2 3.4 0.9 0.5-1.1 0.1 0.4 0.1 0.4v1.4c0 0-14.6 14.1-18.4 20.2-3.7 5.7-4.7 16.6 0.2 19.6 0-0.2-0.2-0.6-0.6-1.2-1.8-2.4-4.3-13 4-22 8.3-9 27.9-27.3 30.7-34.3 0 0-2.4 7.4-5.2 12-2.9 4.5 19.7-23.5 21.7-26.7 0 0-6.2 11.9-8.2 16.5-2 4.6 0.2 8.8 4.4 6.9 4.1-2 6.2-4 6.2-4 0 0-8.2 10.5-10.5 13.9-2.2 3.3-1.2 6.3 3.1 6.7 6.8-2 26.9-1.1 30 12.6 0 0-12.6-11.3-22.3-4.5 0 0 2.5-4.2 4.7-4.8 0 0-12.6-3.8-18.1 5.8-5.4 9.7 3.2 11.8 8.1 11.7 4.8-0.2 21.1-3.8 24.4 9.3 2.7 10.4-7.9 17.6-19 15.5 1.9 3.3 3 6.5 3.6 8.6 1.9 7.1-1.7 12.5-1.7 12.5 1.7 8.3 13.1 6.7 13.1 6.7 0-1.5 3.8-3.1 3.9 2.4 0.1 6.7 5 21.5 5 21.5zm-100.8-100.4c0.7 1.1 0.7 1.7 0.7 1.7l-1.7-1.6 1.6-8.4c1 3.5 2.8 8.6 2.8 8.6-0.5-0.4-4-1.3-3.4-0.3zm25.6 56.6c0.9-4.8 9.6-3.6 9.6-3.6 0.4-8.2 9.5-6.9 9.5-6.9 0 0-1.2-4.1 0.4-7.8 1.6-3.8 7.5-6 7.5-6 0 0-2.3-0.9-1.6-4.1 0.7-5.9 9.2-9.5 9.2-9.5-3.9-8.5 7-15 7-15-4-0.2-3.9-5.6-2.4-7.4 1.5-1.9 7.1-9.6 7.1-9.6-2.7 2.1-7.6 2.2-7.8-2.5-0.2-4.8 5.1-13.1 5.1-13.1-1.7 1.8-16.6 20.1-16.6 20.1 0 2.4 4.4-1.5 4.8-1.9-0.4 0.4-6 5.6-6.7 8.3-0.7 2.8 2.1 3.8 2.1 3.8-2.7 1.2-6.7 5.8-6.3 7.9 0.3 2 4.4 1.8 4.4 1.8-3.1 1.5-6.3 4.8-6.3 6.7 0 1.9 3 2.9 3 2.9-5 1.2-7.1 4.1-7.2 6.2-0.1 2.1 2.5 2.7 2.5 2.7 0 0-2.6 1.2-5.4 2.5-2.8 1.2 0.3 4.4 0.3 4.4-3 0.1-4.2 1.2-4.1 3.1 0.1 2 0.8 2.8 0.8 2.8-4.7-0.9-6 3.3-6 3.3-2.8-2.5-7.8 2-7.8 2 0 0-1.8-4.1-5.9-2.7-2 0.6-3.1 2-3.8 3.4 1.6 2.3 2.7 5.2 3.5 7.7 4.8-4.8 11.1 0.5 11.1 0.5zm31.8-67.3q0 0 0 0 0 0 0 0zm-98.4 27c-2-3.9 1-9.8 1-9.8 0 0-1.8 6.3 1.7 8.1 3.4 1.9 5.5-0.4 5.5-0.4 0 0-1.4 2.7-3.2 2.7 0 0 1.8 0.9 3.5 1 1.7 0.2 6.8-0.7 10.1-2.5 3.4-1.7 10-2 12.6-0.5 0 0-0.9-4.7-4.9-6.2 0 0 2.7-0.6 4.2 0 1.4 0.7 3.7 6.3 3.2 9.1-0.4 2.7-5.7 4.1-6.5 4.1-0.8 0-1.2-0.3-1.2-0.3 0 0 5.4-2.5 5.7-3.9 0.3-1.5-2-2-4.4-2.1-2.3-0.1-6.7 0-9.4 1.4-2.7 1.3-6.9 2.7-10.1 1.9-3.2-0.8-3.9-1.2-3.9-1.2 0 0 1.8 2.7 3 3.6 0 0-5-1.1-6.9-5zm30.2-10.4c0 0-1.7-2.7-4-2.5-2 0.2-1 1.3 0.1 1.3 1.2 0 3.9 1.2 3.9 1.2zm70.4 14.8l-0.1-0.4c-2.3 1.3-6 3.7-7.2 7-1.9 5.1 3.4 5.7 3.4 5.7 0 0-4 0.3-7 2.5q4.7-0.5 8.6 0.1c0.2-0.5 0.8-1.5 2-1.5 0 0 1.9-2.5 4.4-2.5 0 0-1.6 1.2-1.5 2.9 0.2 1.6 1.1 1.3 0.9 2.5 2.3 0.8 4.4 1.9 6.2 3.2 0.1-0.4 0.7-1.5 1.8-1.7 1.3-0.1 1.3-2.5 4.2-1.8 3 0.6 0.2 0.5 0.2 0.5 0 0-1.4 0.8-1.6 1.5-0.3 0.7 0.8 1.5 0.5 2.1-0.3 0.7 0.1 2 2.1 2.3 2 0.3 6.2 0.3 6.2 0.3 0 0-0.4-2.7 1-2.8 1.5-0.2-0.1-2.6 1.6-3.1 0 0 0.2 2.8 1.4 2.7 1.2-0.2 0.9 1.8 0.9 1.8 0 0 3.8-1 3.9-7.1 0 0-1.7 0.3-1.7-0.6 0-1-2-0.8-2-0.8 0 0 0.2-1.2 1.2-1.2 0.9 0 0-1.5 0.8-1.9 0.8-0.4-6.1-5.3-9.1-0.7 0 0-0.3-2.1-2.8-2.1-2.6 0-15.7 1.3-18.3-6.9zm-37 18.2c0.4-0.5 0.5-1 0.5-1.4-0.1 0.3-0.3 0.7-0.5 1.4zm-27.6-28.2c0.1 0.2 0.3 0.4 0.3 0.4 0.7 0.9 2.4-0.5 3.3 0.6 1.2 1.4 5.8-0.4 5.1-1.9-0.7-1.5 1-2 1-2 0 0 0.3 0.1 0.8 0.3-0.5-0.5-0.9-0.8-0.9-0.8 0 0-2.1 0.9-1.8 2.4l-2.8-2.5c0 0 1.5-2.3 0.3-3.5-1.3-1.3-2.4 0.1-3.8 1.1-1.3 1 0.4 4.6 2.9 3l3.1 2.9c0 0-2 1.5-3.9 0.5-1.9-1-2.9 0.2-3.5-0.3zm9 33.4c-2-0.4-4.1-0.5-4.1-0.5l-0.3 0.3q-0.5 0.1-1 0.3c-1.3 0.3-2.2 1-3.3 1.7q-0.6 0.4-1 0.9c-2.4-7.4 1.7-13 5.4-16.2 4-3.3 6.2-7.3 6.2-9.4 0.1-2.1-2.3-3.3-4.8-3.6-2.6-0.4-2.5 0.8-2.5 0.8-1.2-0.8-2.4-2.4-2.4-3.8 0-0.2 0-0.5-0.1-0.8 0 1.7 0.2 4.2 1.9 5 2.6-0.9 6.8-1.1 7.2 2 0.5 3.9-4.9 6.6-4.9 6.6 0 0 2.8-4.2-0.6-3.3-3.4 1-1 3.8-1 3.8-1.6-0.4-1.4-2.6-3.3-1.1-1.9 1.5 0.9 4.3 0.9 4.3-1.9-0.3-2.4-3.1-2.4-3.1 0 0-1.7-0.3-2.2 1.5-0.4 1.9 2.8 4.3 2.8 4.3-2.5-0.1-3.8-3.3-3.8-3.3 0 0-2.1 0.8-1.5 2.9 0.6 2.1 3.3 4.1 3.3 4.1-3.8-0.6-4.5-4.1-4.5-4.1-1.3 1.8-2.5 3.5-0.9 5.9 1.6 2.5 4.6 2.8 4.6 2.8-6.2-0.1-6.4-3.2-6.4-3.2-2.6 9.1 7.6 8.8 7.6 8.8-1.5 0-7.6 0.8-8.8-3.6q0 0.7-0.1 1.8c-0.4 4.3 7.5 6.5 7.5 6.5-4.8 0.5-7.9-2.7-7.9-2.7 0 0-0.7 0.9 0.7 3.7 1.4 2.9 8.2 2.6 8.2 2.6-2.3 1.2-7.2 0-7.2 0 1.8 4.7 7.5 3.8 7.5 3.8-5.9 1.9-7.3-1.4-7.3-1.4 1.8 6 7.1 8.3 9.1 9-0.2-0.8-0.4-1.5-0.8-1.8-1.1-1.1-2.1-1.8-2.1-1.8 0 0 1.6 0.3 3 2.4 0 0-1.8-2.6-1.6-9.5-0.4-5.2 1.6-8.1 1.6-8.1 0 0 3.7-4.9 9.3-4.5zm-9.3 22.1q0 0 0 0 0 0 0 0zm-12 0c0 0-8.4-0.2-18-6.9 0 0 9 7.4 17 7.8zm13.4-46.2q-0.5 0.3-1.1 0.6c0.2 0 0.6-0.1 1.1-0.6zm8.3 47.2c2.6-1.9 2.2-10 2.2-10-0.9 9.4-4.4 10.2-4.4 10.2 0 0 2.4-0.3 1.8 2.8-0.6 3-11.6 13.9-16.3 16.5-4.6 2.6-3.8 4.8-3.8 4.8 0 0 0.9-2.3 5.5-4.7 4.6-2.4 15.7-13.8 16.2-15.6 0.5-1.8-1.2-4-1.2-4zm-6.6 0.6h-0.1 0.1zm16-30.8c2.3 0 6.3-0.3 7.8-0.5-2.4-1.1-4.9-2.4-4.9-3.6 0-2.4-3.2-3.4-3.2-3.4l-1.2-1q0.1-0.5 0.5-1.1c2-3.1 5.3-12.2 2.3-16.5-1.3-1.8-1.2-2.5-1.1-2.8 0.3-0.4 1.1-0.4 1.1-0.4 0.2 0 0.5-0.3 0.5-0.5 0-0.3-0.2-0.6-0.4-0.6-0.1 0-0.6-0.1-1.2-0.1q-0.4 0-0.7 0.1-0.1-1.1-0.1-2.2c-0.6 0.5-1.2 1.5-0.8 2.9v0.1c1-0.5 2.7-0.3 2.7-0.3 0 0-3.4 0-0.5 4.1 3 4.1-0.4 13-2.2 15.9-1.8 2.8 0.3 4.2 0.3 4.2-0.8 0.9-2.1 0.3-2.3-0.4-0.1-0.7-2 2-5 5-2.9 3-0.3 3.2-0.3 3.2 4.8 0.5 6.3-2.1 8.7-2.1zm15.7 9.4q-0.2-0.2-0.6-0.6c0.2 0.2 0.4 0.4 0.6 0.6zm12.7 10.3c-2.4 1.9-4.7 4.4-7.2 5.3-3.6 1.2-10 5.8-15.5 7.1 0.3 0.8 1.4 1.6 3.1 0.7 2.3-1.2 3.4-3 3.4-3-0.5 1.6 1.4 2.4 3.4 1.5 2-0.9 3.4-3.8 3.4-3.8-0.3 1.6 1.2 3 2.6 1.6 1.5-1.4 2.7-4.8 2.7-4.8 0 0 0.2 2.7 2.2 1.2 1.4-1.1 1.8-4.3 1.9-5.8zm3.5-1.9c-1.1 0.3-2.2 0.9-3.2 1.7 0.1 0.5 0.3 1.2 0.4 2.2 0.8-0.3 2.4-1.3 2.8-3.9zm4.7 1q-0.3-0.2-0.6-0.4c-1.1-0.6-2-0.8-3-0.7 0.1 0.6 0 1.3-0.1 1.8-0.7 2.2 2.4 2.6 3.7-0.7zm44 31.6c-0.5-1.7-0.3-4.3-1-6.7-0.8-2.5-2.7-1.2-2.7-1.2 0 0-5.9 0.4-13.3-3.5-7.5-4-9.8-20-12.9-26-3.2-5.9-14.8-5.7-14.8-5.7 0 0 11.6 0.8 13.7 5.9 2.1 5 6.4 19.1 10.6 24.1 4.2 4.9 14.9 6.8 16.6 6.8 1.6 0 2.2-0.9 2.2 1.2 0.1 2.2 1.6 5.1 1.6 5.1zm-39.3-26.2q-0.2-0.6-0.4-1.3c-0.8-1.2-1.9-2.5-3.6-3.7q-0.1 0.4-0.1 0.7c0.1 0.8 1.4 1.6 2.9 2.9 0.4 0.4 0.8 0.9 1.2 1.4z"/>
				</g>
				<g id="Layer">
				</g>
				<g id="Layer">
					<g id="Layer">
						<g id="Layer">
							<path id="Layer" class="s7" d="m-105.2-176.4h725.4v478.8h-725.4z"/>
						</g>
						<g id="Layer">
							<path id="Layer" class="s7" d="m-105.2-176.4h725.4v478.8h-725.4z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m155 51.8q-4.8 0-8.7-2.1-3.9-2.1-6.1-5.7-2.2-3.7-2.2-8.3 0-4.5 2.2-8.2 2.2-3.7 6.1-5.8 3.9-2.1 8.7-2.1 4.8 0 8.6 2.1 3.9 2.1 6.1 5.8 2.1 3.6 2.1 8.2 0 4.6-2.1 8.3-2.2 3.7-6.1 5.7-3.8 2.1-8.6 2.1zm0-4q3.5 0 6.3-1.5 2.8-1.6 4.4-4.4 1.6-2.8 1.6-6.2 0-3.4-1.6-6.2-1.6-2.7-4.4-4.3-2.8-1.6-6.3-1.6-3.5 0-6.4 1.6-2.8 1.6-4.4 4.3-1.6 2.8-1.6 6.2 0 3.4 1.6 6.2 1.6 2.8 4.4 4.4 2.9 1.5 6.4 1.5z"/>
						</g>
						<g id="Layer">
							<path id="Layer" class="s8" d="m195.4 51.8q-4.7 0-8.6-2.1-3.8-2-6-5.7-2.2-3.7-2.2-8.3 0-4.6 2.2-8.2 2.2-3.7 6-5.8 3.9-2.1 8.7-2.1 3.7 0 6.8 1.2 3.1 1.3 5.3 3.7l-3 2.8q-3.6-3.7-8.9-3.7-3.6 0-6.5 1.6-2.8 1.6-4.4 4.3-1.6 2.8-1.6 6.2 0 3.4 1.6 6.2 1.6 2.8 4.4 4.4 2.9 1.5 6.5 1.5 5.3 0 8.9-3.7l3 2.8q-2.2 2.4-5.3 3.7-3.1 1.2-6.9 1.2z"/>
						</g>
						<g id="Layer">
							<path id="Layer" class="s8" d="m239.6 47.6v3.9h-23.1v-31.5h22.4v3.9h-17.9v9.7h16v3.8h-16v10.2z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m269.6 43.6h-16.9l-3.5 7.9h-4.7l14.4-31.5h4.5l14.5 31.5h-4.8zm-1.6-3.6l-6.8-15.5-6.9 15.5z"/>
						</g>
						<g id="Layer">
							<path id="Layer" class="s8" d="m312.4 20v31.5h-3.7l-19.1-23.5v23.5h-4.6v-31.5h3.7l19.1 23.4v-23.4z"/>
						</g>
						<g id="Layer">
							<path id="Layer" class="s8" d="m353.1 51.8q-4.7 0-8.6-2.1-3.8-2-6-5.7-2.2-3.7-2.2-8.3 0-4.6 2.2-8.2 2.2-3.7 6-5.8 3.9-2.1 8.7-2.1 3.7 0 6.8 1.2 3.1 1.3 5.3 3.7l-3 2.8q-3.6-3.7-9-3.7-3.5 0-6.4 1.6-2.8 1.6-4.4 4.3-1.6 2.8-1.6 6.2 0 3.4 1.6 6.2 1.6 2.8 4.4 4.4 2.9 1.5 6.4 1.5 5.4 0 9-3.7l3 2.8q-2.2 2.4-5.3 3.7-3.1 1.2-6.9 1.2z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m388.5 51.8q-4.8 0-8.6-2.1-3.9-2.1-6.1-5.7-2.2-3.7-2.2-8.3 0-4.5 2.2-8.2 2.2-3.7 6.1-5.8 3.8-2.1 8.6-2.1 4.8 0 8.7 2.1 3.8 2.1 6 5.8 2.2 3.6 2.2 8.2 0 4.6-2.2 8.3-2.2 3.7-6 5.7-3.9 2.1-8.7 2.1zm0-4q3.5 0 6.3-1.5 2.8-1.6 4.4-4.4 1.7-2.8 1.7-6.2 0-3.4-1.7-6.2-1.6-2.7-4.4-4.3-2.8-1.6-6.3-1.6-3.5 0-6.3 1.6-2.9 1.6-4.5 4.3-1.6 2.8-1.6 6.2 0 3.4 1.6 6.2 1.6 2.8 4.5 4.4 2.8 1.5 6.3 1.5z"/>
						</g>
						<g id="Layer">
							<path id="Layer" class="s8" d="m414.7 20h4.6v27.6h17.2v3.9h-21.8z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m458.5 51.8q-4.8 0-8.7-2.1-3.8-2.1-6-5.7-2.2-3.7-2.2-8.3 0-4.5 2.2-8.2 2.2-3.7 6-5.8 3.9-2.1 8.7-2.1 4.8 0 8.7 2.1 3.8 2.1 6 5.8 2.2 3.6 2.2 8.2 0 4.6-2.2 8.3-2.2 3.7-6 5.7-3.9 2.1-8.7 2.1zm0-4q3.5 0 6.3-1.5 2.8-1.6 4.4-4.4 1.6-2.8 1.6-6.2 0-3.4-1.6-6.2-1.6-2.7-4.4-4.3-2.8-1.6-6.3-1.6-3.5 0-6.3 1.6-2.9 1.6-4.5 4.3-1.6 2.8-1.6 6.2 0 3.4 1.6 6.2 1.6 2.8 4.5 4.4 2.8 1.5 6.3 1.5z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m506 51.5l-6.8-9.7q-1.3 0.1-2 0.1h-7.9v9.6h-4.6v-31.5h12.5q6.1 0 9.7 2.9 3.5 2.9 3.5 8 0 3.7-1.8 6.3-1.8 2.5-5.2 3.7l7.6 10.6zm-9-13.4q4.3 0 6.6-1.9 2.3-1.8 2.3-5.3 0-3.4-2.3-5.2-2.3-1.8-6.6-1.8h-7.7v14.2z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m192 70.9q2.4-2.4 6.1-2.4 3.6 0 6.1 2.4 2.5 2.3 2.5 5.8 0 3.4-2.5 5.8-2.5 2.3-6.1 2.3-3.7 0-6.1-2.3-2.5-2.4-2.5-5.8 0-3.5 2.5-5.8zm6.1 0.1q-2.4 0-4.1 1.6-1.8 1.6-1.7 4-0.1 2.4 1.7 4.1 1.7 1.7 4.1 1.6 2.4 0.1 4.1-1.6 1.7-1.7 1.7-4.1 0-2.4-1.7-4-1.7-1.6-4.1-1.6z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m213.9 68.6h7q2.6 0 4 1.1 1.5 1 1.5 2.9 0 1.4-0.8 2.3-0.8 1-2.1 1.3 1.6 0.3 2.6 1.4 0.9 1 0.9 2.7 0 2-1.6 3.2-1.6 1.2-4.3 1.2h-7.2zm2.7 2.4v4.2h4.2q1.3 0 2-0.5 0.8-0.6 0.8-1.6 0-1-0.8-1.6-0.7-0.5-2-0.5zm0 6.6v4.6h4.2q1.6 0 2.5-0.6 0.9-0.6 0.9-1.7 0-1.1-0.9-1.7-0.9-0.6-2.5-0.6z"/>
						</g>
						<g id="Layer">
							<path id="Layer" class="s8" d="m233.9 82.3q0.4-0.4 1-0.4 0.6 0 1 0.4 0.5 0.4 0.5 1.1 0 0.6-0.5 1.1-0.4 0.4-1 0.4-0.6 0-1-0.4-0.5-0.5-0.4-1.1-0.1-0.7 0.4-1.1z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m243.8 68.6h6.7q3.6 0 6 2.3 2.4 2.3 2.4 5.7 0 3.5-2.4 5.8-2.4 2.3-6.1 2.3h-6.6zm2.8 2.4v11.2h3.9q2.4 0 4-1.6 1.6-1.5 1.6-4 0-2.4-1.6-4-1.7-1.6-4.1-1.6z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m276.5 81h-8.1l-1.6 3.7h-2.8l7.1-16.1h2.8l7.1 16.1h-2.9zm-1.1-2.4l-3-7.2-3.1 7.2z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m297.9 81h-8.1l-1.6 3.7h-2.9l7.2-16.1h2.8l7.1 16.1h-2.9zm-1.1-2.4l-3-7.2-3.1 7.2z"/>
						</g>
						<g id="Layer">
							<path id="Layer" class="s8" d="m316 68.5q1.8 0 3.4 0.7 1.6 0.6 2.8 1.8l-1.6 2q-0.9-1-2.1-1.5-1.2-0.6-2.4-0.6-2.5 0-4.1 1.7-1.7 1.6-1.7 4 0 2.4 1.7 4 1.6 1.7 4.1 1.7 2.5 0 4.5-1.9l1.6 1.7q-1.2 1.3-2.9 2-1.7 0.7-3.4 0.7-3.5 0-6-2.3-2.4-2.4-2.4-5.9 0-3.4 2.5-5.7 2.4-2.4 6-2.4z"/>
						</g>
						<g id="Layer">
							<path id="Layer" class="s9" d="m352.5 87.2h-2.9v-20.9h2.9z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m382.5 70.9q2.5-2.4 6.1-2.4 3.7 0 6.1 2.4 2.5 2.3 2.5 5.8 0 3.4-2.5 5.8-2.4 2.3-6.1 2.3-3.6 0-6.1-2.3-2.5-2.4-2.5-5.8 0-3.5 2.5-5.8zm6.2 0.1q-2.5 0-4.2 1.6-1.7 1.6-1.7 4 0 2.4 1.7 4.1 1.8 1.7 4.1 1.6 2.4 0.1 4.1-1.6 1.7-1.7 1.7-4.1 0-2.4-1.7-4-1.7-1.6-4-1.6z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m404.4 68.6h7.1q2.5 0 3.9 1.1 1.5 1 1.5 2.9 0 1.4-0.8 2.3-0.7 1-2 1.3 1.6 0.3 2.5 1.4 0.9 1 0.9 2.7 0 2-1.5 3.2-1.6 1.2-4.4 1.2h-7.2zm2.8 2.4v4.2h4.1q1.3 0 2.1-0.5 0.7-0.6 0.7-1.6 0-1-0.7-1.6-0.7-0.5-2.1-0.5zm0 6.6v4.6h4.2q1.6 0 2.5-0.6 0.9-0.6 0.9-1.7 0-1.1-0.9-1.7-0.9-0.6-2.5-0.6z"/>
						</g>
						<g id="Layer">
							<path id="Layer" fill-rule="evenodd" class="s8" d="m425.1 68.6h6.4q3.1 0 4.8 1.5 1.7 1.4 1.7 4 0 2.7-1.7 4.2-1.7 1.5-4.8 1.5h-3.7v4.9h-2.7zm2.7 2.4v6.4h3.6q3.9 0 3.9-3.3 0-3.1-3.9-3.1z"/>
						</g>
						<g id="Layer">
							<path id="Layer" class="s8" d="m456 76.8h2.4v5.9q-1.2 0.9-2.9 1.5-1.7 0.6-3.4 0.6-3.6 0-6-2.3-2.5-2.4-2.5-5.8 0-3.5 2.5-5.8 2.5-2.4 6.2-2.4 1.7 0 3.4 0.7 1.6 0.6 2.8 1.6l-1.5 2q-2-1.9-4.7-1.9-2.5 0-4.2 1.7-1.7 1.6-1.7 4 0 2.4 1.7 4.1 1.7 1.7 4.2 1.6 1.9 0.1 3.7-1.1z"/>
						</g>
						<g id="Clip-Path" clip-path="url(#cp1)">
							<g id="Layer">
								<path id="Layer" class="s9" d="m43.6 50.4c3.9-3.1 6.6-7.4 9.8-11.2 3.2-3.8 7.3-7.3 12.3-7.8 4.2-0.3 8.2 1.6 11.8 3.8 3.5 2.2 6.9 4.8 10.8 6.2 6.1 2.2 13.1 1 18.6-2.4-6.9-21.5-27.3-37.1-51.4-37.1-26 0-47.8 18.2-52.8 42.5 3 2.6 6.3 5 9.8 6.9 9.7 5 22.5 5.9 31.1-0.9z"/>
							</g>
						</g>
						<g id="Clip-Path" clip-path="url(#cp2)">
							<g id="Layer">
								<path id="Layer" class="s10" d="m84.7 73.3c-3.9-1.1-7.8-3.1-11.9-2.9-6.4 0.4-11.3 5.9-13.9 11.7-2.6 5.7-3.8 12.1-7 17.6-1.8 3.1-4.2 5.8-7 8q5.2 1 10.7 1c24.5 0 45.3-16.3 51.8-38.5-6.8 3.9-15.1 5.3-22.7 3.1z"/>
							</g>
						</g>
						<g id="Clip-Path" clip-path="url(#cp3)">
							<g id="Layer">
								<path id="Layer" class="s11" d="m48.8 89.9c3.3-3.8 5-8.5 7.4-12.9 2.4-4.3 5.8-8.6 10.6-10 4-1.1 8.3-0.1 12.3 1.4 3.9 1.5 7.7 3.4 11.8 4.1 6.2 0.9 12.5-1.4 17.2-5.4 0.9-3.8 1.3-7.8 1.3-11.8 0-4.8-0.6-9.6-1.8-14-8 3.6-17.7 3.8-25.6-0.2-3.6-1.9-7-4.6-11.1-5.1-6.3-0.9-12.2 3.5-15.9 8.7-3.7 5.1-6.1 11.2-10.3 15.9-7.4 8.3-19.4 11.4-30.5 10.7-3.6-0.2-7.2-0.8-10.7-1.7 3.2 11.3 10.1 21.1 19.2 28.1 9.5 1.6 20-0.6 26.1-7.8z"/>
							</g>
						</g>
					</g>
				</g>
			</svg>		
		
		</a></div>
		<div class="oc-header-item"><img class="oc-nasa-logo-image" src="https://cdn.earthdata.nasa.gov/eui/latest/docs/assets/ed-logos/meatball_hover_2x.png" alt="NASA Logo" /></div>

	</div>

</div>

<div class="full-width-container">

<div class="navbar navbar-default" role="navigation">
  <div class="container">
    <div class="navbar-header">
      <button type="button" class="navbar-toggle" data-toggle="collapse" data-target=".navbar-collapse">
        <span class="sr-only">Toggle navigation</span>
        <span class="icon-bar"></span>
        <span class="icon-bar"></span>
        <span class="icon-bar"></span>
      </button>
    </div>
    <div class="navbar-collapse collapse">

		<div id="ocean_color_navbar"></div>
		<div id="quicklinks_navbar"></div>

    </div><!--/.nav-collapse -->
  </div><!--/.container -->
</div>

</div>

<script>
  let quickLinksNav = document.querySelector("#quicklinks_navbar");
  quickLinksNav.addEventListener('click', function(event) {
    event.stopPropagation();
    quickLinksNav.classList.add("display-dropdown");
  })

  let oceanColorNav = document.querySelector("#ocean_color_navbar");
  oceanColorNav.addEventListener('click', function(event) {
    event.stopPropagation();
    oceanColorNav.classList.add("display-dropdown");
  })

</script>

<div class="full-width-container">


   <div class="inside_page_header_image" id="background">
      <div class="inside_page_header_text eui-hero__title">Ocean Data</div>
      <div class="inside_page_subheader_text eui-hero__subtitle"></div>
   </div>


   <div class="content-full">

       
<form name="getfile_form" id="getfile_form">
   <table class="table table-responsive">
      <tr>
            <td id="generateText" class="eui-banner--success">
               The GetFile Application<br>
            </td>
      </tr>
      
      <tr>
         <td class="eui-banner--danger">
            Invalid appkey/token used.<br>
         </td>
      </tr>
      
   </table>
</form>



   </div>

</div>


<div class="full-width-container">
	<div class="footer-grid">
		<div class="footer-item">
			<div class="footer-title">ABOUT</div>
			<div class="footer-links">
				<!--<p><a href="" target="_blank">What We Do</a></p>-->
				<p><a href="https://oceancolor.gsfc.nasa.gov/about/staff/">Our Team</a></p>
				<p><a href="https://oceancolor.gsfc.nasa.gov/about/contact/">Contact Us</a></p>
			</div>

			<div class="footer-title" style="margin-top: 40px;">DATA</div>
			<div class="footer-links">
				<p><a href="https://oceancolor.gsfc.nasa.gov/data/getting-started/">Get Started</a></p>
				<p><a href="https://oceancolor.gsfc.nasa.gov/data/find-data/">Find Data</a></p>
				<p><a href="https://oceancolor.gsfc.nasa.gov/data/use-data/">Use Data</a></p>
			</div>

			
		</div>
		<div class="footer-item">
			<div class="footer-title">TOOLS</div>
			<div class="footer-links">
				<p><a href="https://seabass.gsfc.nasa.gov/" target="_blank">SeaBASS</a></p>
				<p><a href="https://seadas.gsfc.nasa.gov/" target="_blank">SeaDAS</a></p>
				<p><a href="https://giovanni.gsfc.nasa.gov/giovanni/" class="ext" target="_blank">Giovanni Data Visualizer</a></p>
				<p><a href="https://worldview.earthdata.nasa.gov/" class="ext" target="_blank">Worldview Mapping Interface</a></p>
				<p><a href="https://oceandata.sci.gsfc.nasa.gov/mqm/" target="_blank">Mission Quality Monitor</a></p>
			</div>

			

		</div>
		<div class="footer-item">
			<div class="footer-title">HELP</div>
			<div class="footer-links">
				<p><a href="https://www.youtube.com/watch?v=QtfMlkd7kII" target="_blank">How to Use Earthdata Search</a></p>
				<p><a href="https://oceancolor.gsfc.nasa.gov/l3/help/" target="_blank">Level 3 &amp; 4 Browser Help</a></p>
				<p><a href="https://oceandata.sci.gsfc.nasa.gov/api/file_search_help/" target="_blank">File Search Help</a></p>
				<p><a href="https://oceancolor.gsfc.nasa.gov/data/download_methods/" target="_blank">Search & Download Help</a></p>
				<p><a href="https://forum.earthdata.nasa.gov/viewforum.php?f=7&&DAAC=86&sid=9c4bae2136e93fa40cbf1f73f254ac3e" class="ext" target="_blank">Earthdata Forum</a></p>
				<p><a href="https://oceancolor.gsfc.nasa.gov/resources/mailing-lists/" target="_blank">Join Mailing List</a></p>
				<p><a href="https://oceancolor.gsfc.nasa.gov/resources/how-to-cite/">How to Cite Data</a></p>
			</div>
			

		</div>
		<div class="footer-item">

			<div class="footer-title">CONNECT WITH US</div>
			<div class="social-media-icons">
				<a href="https://twitter.com/nasaearth" aria-label="View the Twitter page for NASA Earth" target="_blank"><i class="fa-brands fa-square-x-twitter" style="font-size: 2em; color: #dbdbdb;"></i></a> <a href="https://www.facebook.com/nasaearth" aria-label="View the Facebook page for NASA Earth" target="_blank"><i class="fa-brands fa-square-facebook" style="font-size: 2em; color: #dbdbdb;"></i></a> <a href="https://www.instagram.com/nasaearth/" aria-label="View the Instagram page for NASA Earth" target="_blank"><i class="fa-brands fa-square-instagram" style="font-size: 2em; color: #dbdbdb;"></i></a>
			</div>

			<div class="footer-title" style="margin-top: 40px;">COMPLIANCE</div>
			<div class="footer-links"> 
				<p><a class="ext" href="https://www.nasa.gov/about/highlights/HP_Privacy.html">Web Privacy Policy</a></p>
				<p><a class="ext" href="https://science.nasa.gov/researchers/science-information-policy/">Data &amp; Information Policy</a></p>
				<p><a class="ext" href="https://www.nasa.gov/audience/formedia/features/communication_policy.html">Communications Policy</a></p>
				<p><a class="ext" href="https://www.nasa.gov/accessibility/">Accessibility</a></p>
				<p><a class="ext" href="https://www.nasa.gov/FOIA/index.html">Freedom of Information Act</a></p>
				<p><a class="ext" href="https://www.usa.gov/">USA.gov</a></p>
			</div>

			
		</div>
		
		<div class="footer-item">
			<div class="responsible-party-grid">
				<div class="responsible-party-item">
               
               <a title="World Data System" href="https://worlddatasystem.org/"><img src="https://oceancolor.gsfc.nasa.gov/images/wds_logo-1.png" style="width: 55px; height: auto; background: white;"></a>
               <a title="CoreTrustSeal" href="https://www.coretrustseal.org/"><img src="https://oceancolor.gsfc.nasa.gov/images/CoreTrustSeal-logo.jpg" style="width: 60px; height: auto; border: none;"></a>
   
            </div>
				<div class="responsible-party-item">
               Responsible NASA Official: <a href="https://science.gsfc.nasa.gov/sed/bio/sean.w.bailey" target="blank" style="color: #fafafa;">Sean Bailey</a>
               <br>
               Curator: <a href="mailto:webadmin@oceancolor.gsfc.nasa.gov" style="color:#fafafa;">OceanColor Webmaster</a>
            </div>
			</div>
		</div>
	</div>
</div>

</body>

</html>
