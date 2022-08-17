#include "cjitter.h"

static inline double image_hue2rgb(double p, double q, double t) {
	if (t < 0.) t += 1;
	if (t > 1.) t -= 1;
	if (t < 1./6)
		return p + (q - p) * 6. * t;
	else if (t < 1./2)
		return q;
	else if (t < 2./3)
		return p + (q - p) * (2./3 - t) * 6.;
	else
		return p;
}

/*
This function randomly changes the color in each channel in HSL space. It first
converts the RGB batch of images to HSL; then applies a different color
jittering in each channel; and converts back to RGB space. The function expects
a set of images of size [3,Nim,H,W] in the RGB range [0, 255]. It does this
color jittering operation in place.
*/
void cjitter(float* img, int height, int width, float h_change, float s_change, float l_change){
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(-1, 1);
	
	h_change *= dist(gen);
	s_change *= dist(gen);
	l_change *= dist(gen);

	// std::cout<<"h_change : "<<h_change<<std::endl;
	// std::cout<<"s_change : "<<s_change<<std::endl;
	// std::cout<<"l_change : "<<l_change<<std::endl;

	float *r_ptr = img;
	float *g_ptr = r_ptr + height * width;
	float *b_ptr = g_ptr + height * width;

	double r,g,b, h,s,l;
	for (long pixel = 0; pixel < height*width; ++pixel) {
		
		// tic = timenano();
		// load_t += (timenano() - tic);

		/**************** First convert to HSL ****************/
		// tic = timenano();
		// get RGB
		r = ((double)*r_ptr) / 255;
		g = ((double)*g_ptr) / 255;
		b = ((double)*b_ptr) / 255;

		// std::cout<<"Iteration : ("<<pixel1<<","<<pixel2<< ") " << r<< " " << g << " " << b << std::endl;

		double mx = max(max(r, g), b);
		double mn = min(min(r, g), b);
		if(mx == mn) {
			h = 0; // achromatic
			s = 0;
			l = mx;
		} else {
			double d = mx - mn;
			if (mx == r) {
				h = (g - b) / d + (g < b ? 6 : 0);
			} else if (mx == g) {
				h = (b - r) / d + 2;
			} else {
				h = (r - g) / d + 4;
			}
			h /= 6;
			l = (mx + mn) / 2;
			s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
		}

		// rgb2hsl_t += (timenano() - tic);

		/**************** Change color by *_change ****************/
		// tic = timenano();

		h += h_change;
		h = fmod(h, 1.0);
		h = h < 0 ? 1 + h : h;
		s += s_change;
		s = s < 0 ? 0 : (s > 1 ? 1 : s);
		l += l_change;
		l = l < 0 ? 0 : (l > 1 ? 1 : l);

		// clrchg_t += (timenano() - tic);

		/**************** Finally convert back to RGB ****************/
		// tic = timenano();

		if(s == 0) {
			// achromatic
			r = l;
			g = l;
			b = l;
		} else {
			double q = (l < 0.5) ? (l * (1 + s)) : (l + s - l * s);
			double p = 2 * l - q;
			double hr = h + 1./3;
			double hg = h;
			double hb = h - 1./3;
			r = image_hue2rgb(p, q, hr);
			g = image_hue2rgb(p, q, hg);
			b = image_hue2rgb(p, q, hb);
		}

		// hsl2rgb_t += (timenano() - tic);

		/**************** Clamp and store values ****************/
		// tic = timenano();

		r *= 255;
		g *= 255;
		b *= 255;
		*r_ptr = r;
		*g_ptr = g;
		*b_ptr = b;

		++r_ptr;
		++g_ptr;
		++b_ptr;

		// std::cout<<"Iteration : ("<<pixel1<<","<<pixel2<< ") " << r << " " << g << " " << b << std::endl;
	}
}
