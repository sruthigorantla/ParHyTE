#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<algorithm>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<omp.h>
using namespace std;

bool debug=false;

string version;
string trainortest = "test";
int relation_num,entity_num,time_num;
int m= 100; //relation dim
int n= 100; //entity dim
string dataset = "";
int nthread = 1;
string resultpath="";

map<string,int> relation2id,entity2id,time2id;
map<int,string> id2entity,id2relation,id2time;
map<string,string> mid2name,mid2type;
map<int,map<int,int> > entity2num;
map<int,int > e2num;
map<pair<string,string>,map<string,double> > rel_left,rel_right;
vector<vector<double> > A;
vector<vector<double> > relation_vec,entity_vec;

map<pair<int,int>, map<int,int> > ok;

string Int_to_String(int n)
{
    ostringstream stream;
    stream<<n;  //n为int类型
    return stream.str();
}
int String_to_Int(string s)
{
    stringstream ss;
    ss<<s;
    int num;
    ss>>num;
    return num;
}

double String_to_Double(string s)
{
    stringstream ss;
    ss<<s;
    double num;
    ss>>num;
    return num;
}

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
	double res=0;
	for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	return sqrt(res);
}

void vec_output(vector<double> a)
{
	for (int i=0; i<a.size(); i++)
	{
		cout<<a[i]<<"\t";
		if (i%5==4)
			cout<<endl;
	}
	cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
	return a.second<b.second;
}

class Test{

    vector<int> h,l,r,t;
     vector<int> fb_h,fb_l,fb_r,fb_t;
    vector<vector<int> > feature;
    
    
    double res ;
public:
    void add(int x,int y,int z,int tim, bool flag)
    {
    	if (flag)
    	{
        	fb_h.push_back(x);
        	fb_r.push_back(z);
        	fb_l.push_back(y);
		fb_t.push_back(tim);
        }
        ok[make_pair(x,z)][y]=1;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;
    
    double calc_sum(int e1,int e2,int rel,int tim,vector<vector<double> > &relation_vec,vector<vector<double> >&entity_vec,vector<vector<double> > &A)
    {
        double tmp1=0,tmp2=0,tmp3=0;
        for (int jj=0; jj<n; jj++)
        {
            tmp1+=A[tim][jj]*entity_vec[e1][jj];
            tmp2+=A[tim][jj]*entity_vec[e2][jj];
            tmp3+=A[tim][jj]*relation_vec[rel][jj];
        }

        double sum=0;
        for (int ii=0; ii<n; ii++)
            sum+=-fabs(entity_vec[e2][ii]-tmp2*A[tim][ii]-(entity_vec[e1][ii]-tmp1*A[tim][ii])-(relation_vec[rel][ii]-tmp3*A[tim][ii]));
        return sum;
    }
    void run()
    {
        FILE* f1 = fopen((resultpath+"/relation2vec."+version).c_str(),"r");
        FILE* f3 = fopen((resultpath+"/entity2vec."+version).c_str(),"r");
        FILE* f5 = fopen((resultpath+"/A."+version).c_str(),"r");

        if(!f1)
        {
            printf("can't open relation2vec.");
            exit(-1);
        }
        if(!f3)
        {
            printf("can't open entity2vec.");
            exit(-1);
        }
        if(!f5)
        {
            printf("can't open /A.");
            exit(-1);
        }
        
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb);
        for (int i=0; i<relation_num_fb;i++)
        {
            relation_vec[i].resize(m);
            for (int ii=0; ii<m; ii++)
                fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_num;i++)
        {
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f3,"%lf",&entity_vec[i][ii]);
        }
        A.resize(time_num);
        for (int i=0; i<time_num; i++)
        {
		    A[i].resize(n);
		    for (int jj=0; jj<n; jj++)
		    {
                fscanf(f5,"%lf",&A[i][jj]);
		    }
		}
        fclose(f1);
        fclose(f3);
        fclose(f5);

		double lsum=0 ,lsum_filter= 0;
		double rsum = 0,rsum_filter=0;
		double lp_n=0,lp_n_filter;
		double rp_n=0,rp_n_filter;
        double mid_sum = 0, mid_sum_filter=0;
        double mid_p_n=0,mid_p_n_filter=0;
        int relation_hit_n = 10;

		map<int,double> lsum_r,lsum_filter_r;
		map<int,double> rsum_r,rsum_filter_r;
		map<int,double> lp_n_r,lp_n_filter_r;
		map<int,double> rp_n_r,rp_n_filter_r;
		map<int,int> rel_num;

        #pragma omp parallel for firstprivate(ok,A,entity_vec,relation_vec) reduction(+:lsum,rsum,lp_n,rp_n,lsum_filter,rsum_filter,lp_n_filter,rp_n_filter)
        for (int testid = 0; testid<fb_l.size(); testid+=1)
        {
            int h = fb_h[testid];
            int l = fb_l[testid];
            int rel = fb_r[testid];
	    int tim = fb_t[testid];
            
            vector<pair<int,double> > a;
            for (int i=0; i<entity_num; i++)
            {
                double sum = calc_sum(i,l,rel,tim,relation_vec,entity_vec,A);
                a.push_back(make_pair(i,sum));
            }
            sort(a.begin(),a.end(),cmp);
            double ttt=0;
            int filter = 0;
            for (int i=a.size()-1; i>=0; i--)
            {

                if(ok.find(make_pair(a[i].first,rel))!=ok.end() && ok[make_pair(a[i].first,rel)].count(l)>0)
                    ttt++;
                if(!(ok.find(make_pair(a[i].first,rel))!=ok.end() && ok[make_pair(a[i].first,rel)].count(l)>0))
                    filter+=1;  

                if (a[i].first ==h)
                {
                    lsum+=a.size()-i;
                    lsum_filter+=filter+1;
                    // lsum_r[rel]+=a.size()-i;
                    // lsum_filter_r[rel]+=filter+1;
                    if (a.size()-i<=10)
                    {
                        lp_n+=1;
                        // lp_n_r[rel]+=1;
                    }
                    if (filter<100)
                    {
                        lp_n_filter+=1;
                        // lp_n_filter_r[rel]+=1;
                    }
                    break;
                }
            }
            a.clear();
            for (int i=0; i<entity_num; i++)
            {
                double sum = calc_sum(h,i,rel,tim,relation_vec,entity_vec,A);
                a.push_back(make_pair(i,sum));
            }
            sort(a.begin(),a.end(),cmp);
            ttt=0;
            filter=0;
            for (int i=a.size()-1; i>=0; i--)
            {

                if(ok.find(make_pair(h,rel))!=ok.end() && ok[make_pair(h,rel)].count(a[i].first)>0)
                    ttt++;
                if(!(ok.find(make_pair(h,rel))!=ok.end() && ok[make_pair(h,rel)].count(a[i].first)>0))
                    filter+=1;

                if (a[i].first==l)
                {
                    rsum+=a.size()-i;
                    rsum_filter+=filter+1;
                    // rsum_r[rel]+=a.size()-i;
                    // rsum_filter_r[rel]+=filter+1;
                    if (a.size()-i<=10)
                    {
                        rp_n+=1;
                        // rp_n_r[rel]+=1;
                    }
                    if (filter<100)
                    {
                        rp_n_filter+=1;
                        // rp_n_filter_r[rel]+=1;
                    }
                    break;
                }
            }

            a.clear();
            for(int i=0;i<relation_num;i++)
            {
                double sum = 0;
                sum += calc_sum(h,l,i,tim,relation_vec,entity_vec,A);
                a.push_back(make_pair(i,sum));
            }
            sort(a.begin(),a.end(),cmp);
            filter = 0;
            for(int i=a.size()-1;i>=0;i--)
            {
                if(ok[make_pair(h,a[i].first)].count(l)==0)
                    filter += 1;
                if(a[i].first==rel)
                {
                    mid_sum += a.size()-i;
                    mid_sum_filter += filter+1;
                    // mid_sum_r[rel]+=a.size()-i;
                    // mid_sum_filter_r[rel]+=filter+1;
                    if(a.size()-i<=relation_hit_n)
                    {
                        mid_p_n += 1;
                        // mid_p_n_r[rel]+=1;
                    }
                    if(filter<relation_hit_n)
                    {
                        mid_p_n_filter += 1;
                        // mid_p_n_filter_r[rel]+=1;
                    }
                    break;
                }
            }

        //  break;
        //  if (testid%100==0)
        //  cout<<testid<<":"<<"\t"<<lsum/(testid+1)<<' '<<lp_n/(testid+1)<<' '<<rsum/(testid+1)<<' '<<rp_n/(testid+1)<<"\t"<<lsum_filter/(testid+1)<<' '<<lp_n_filter/(testid+1)<<' '<<rsum_filter/(testid+1)<<' '<<rp_n_filter/(testid+1)<<endl;
        }

		cout<<"left:"<<lsum/fb_h.size()<<'\t'<<lp_n/fb_h.size()<<"\t"<<lsum_filter/fb_h.size()<<'\t'<<lp_n_filter/fb_h.size()<<endl;
		cout<<"right:"<<rsum/fb_l.size()<<'\t'<<rp_n/fb_l.size()<<'\t'<<rsum_filter/fb_l.size()<<'\t'<<rp_n_filter/fb_l.size()<<endl;
        cout<<"relation: "<<mid_sum/(fb_r.size())<<' '<<mid_p_n/(fb_r.size())<<"\t"<<mid_sum_filter/(fb_r.size())<<' '<<mid_p_n_filter/(fb_r.size())<<endl;

        FILE* f = fopen("./all.TransH.log","a");
        fprintf(f, "left:%f\t%f\t%f\t%f\n",lsum/fb_h.size(),lp_n/fb_h.size(), lsum_filter/fb_h.size(),lp_n_filter/fb_h.size());
        fprintf(f, "right:%f\t%f\t%f\t%f\n\n",rsum/fb_l.size(),rp_n/fb_l.size(), rsum_filter/fb_l.size(),rp_n_filter/fb_l.size());
        fprintf(f, "relation:%f\t%f\t%f\t%f\n\n",mid_sum/(fb_r.size()),mid_p_n/(fb_r.size()), mid_sum_filter/(fb_r.size()),mid_p_n_filter/(fb_r.size()));

        fclose(f);


  //       for (int rel=0; rel<relation_num; rel++)
		// {
		// 	int num = rel_num[rel];
		// 	cout<<"rel:"<<id2relation[rel]<<' '<<num<<endl;
		// 	cout<<"left:"<<lsum_r[rel]/num<<'\t'<<lp_n_r[rel]/num<<"\t"<<lsum_filter_r[rel]/num<<'\t'<<lp_n_filter_r[rel]/num<<endl;
		// 	cout<<"right:"<<rsum_r[rel]/num<<'\t'<<rp_n_r[rel]/num<<'\t'<<rsum_filter_r[rel]/num<<'\t'<<rp_n_filter_r[rel]/num<<endl;
		// }
    }

};
Test test;

void prepare()
{
 //    FILE* f1 = fopen("../data/family/entity2id.txt","r");
	// FILE* f2 = fopen("../data/family/relation2id.txt","r");
    FILE* f1 = fopen(("./"+dataset+"/entity2id.txt").c_str(),"r");
    FILE* f2 = fopen(("./"+dataset+"/relation2id.txt").c_str(),"r");
    FILE* f3 = fopen(("./"+dataset+"/time2id.txt").c_str(),"r");
    if(!f1)
    {
        printf("can't open /entity2id.txt");
        exit(-1);
    } 
    if(!f2)
    {
        printf("can't open /relation2id.txt");
        exit(-1);  
    } 
	int x;
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;
		id2entity[x]=st;
		mid2type[st]="None";
		entity_num++;
	}
    int y;
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
	while (fscanf(f3,"%s%d",buf,&x)==2)
	{
		string st=buf;
		time2id[st]=x;
		id2time[x]=st;
		time_num++;
	}

    FILE* f_kb = fopen(("./"+dataset+"/test.txt").c_str(),"r");
    if(!f_kb)
    {
        printf("can't open ./test.txt");
        exit(-1);
    }

	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;
        fscanf(f_kb,"%s",buf);
        string s3=buf;
        fscanf(f_kb,"%s",buf);
        string s4=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        if (time2id.count(s4)==0)
        {
            time2id[s3] = time_num;
            time_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],time2id[s4],true);
    }
    fclose(f_kb);
    FILE* f_kb1 = fopen(("./"+dataset+"/train.txt").c_str(),"r");
    if(!f_kb1)
    {
        printf("can't open ./train.txt");
        exit(-1);
    }


	while (fscanf(f_kb1,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb1,"%s",buf);
        string s2=buf;
        fscanf(f_kb1,"%s",buf);
        string s3=buf;
        fscanf(f_kb1,"%s",buf);
        string s4=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        if (time2id.count(s4)==0)
        {
            time2id[s3] = time_num;
            time_num++;
        }
        entity2num[relation2id[s3]][entity2id[s1]]+=1;
        entity2num[relation2id[s3]][entity2id[s2]]+=1;
        e2num[entity2id[s1]]+=1;
        e2num[entity2id[s2]]+=1;
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],time2id[s4],false);
    }
    fclose(f_kb1);
    FILE* f_kb2 = fopen(("./"+dataset+"/valid.txt").c_str(),"r");
    if(!f_kb2)
    {
        printf("can't open valid.txt");
        exit(-1);
    }
	while (fscanf(f_kb2,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb2,"%s",buf);
        string s2=buf;
        fscanf(f_kb2,"%s",buf);
        string s3=buf;
        fscanf(f_kb2,"%s",buf);
        string s4=buf;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        if (time2id.count(s4)==0)
        {
            time2id[s3] = time_num;
            time_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],time2id[s4],false);
    }
    fclose(f_kb2);
}


int main(int argc,char**argv)
{
    double rate = 0.001;
    double margin = 1;
    int method = 0;
    int nepoch;
    string strategy;

    if (argc<2)
        return 0;
    else
    {
        version = argv[1];
        nthread = String_to_Int(argv[2]);
        dataset = argv[3];
        n=String_to_Int(argv[4]);
        m=n;
        margin=String_to_Double(argv[5]);
        nepoch=String_to_Int(argv[6]);
        resultpath=argv[7];
        rate= String_to_Double(argv[8]);
        strategy = argv[9];

        cout<<"strategy:"<<strategy<<" dataset:"<<dataset<<" nthread:"<<nthread<<" k:"<<n<<" margin:"<<margin<<" nepoch:"<<nepoch<<" rate:"<<rate<<" version:"<<version<<endl;

        time_t start,stop;
        start = time(NULL);
        prepare();
        stop = time(NULL);
        printf("Prepare Time:%ld\n",(stop-start));

        FILE* f = fopen("./all.HyTE.log","a");
        fprintf(f,"strategy:%s dataset:%s version:%s nthread:%d k:%d margin:%f nepoch:%d rate:%f \n",strategy.c_str(),dataset.c_str(),version.c_str(),nthread,n,margin,nepoch,rate);
        fclose(f);
        start=time(NULL);
        test.run();
        stop = time(NULL);
        printf("Test Time:%ld\n",(stop-start));

    }
}

