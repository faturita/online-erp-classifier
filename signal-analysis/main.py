import joblib
import trainer
import tester
import utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Get sum(abs(hits - no hits)) values and graphs
# trainer.get_epochs_data_list_hit_no_hit(
   # ['../../../data/subject1/subject1-1/record-bv-generic-alex-[2019.06.15-18.43.41].vhdr',
   #   '../../../data/subject1/subject1-2/record-bv-generic-alex-[2019.06.15-19.04.43].vhdr',
   # '../../../data/subject2/subject2-1/record-bv-generic-gonza-[2019.05.25-14.37.56].vhdr',
   # '../../../data/subject2/subject2-2/record-bv-generic-gonza-[2019.05.25-14.43.22].vhdr',
   #  '../../../data/subject3/subject3-1/record-bv-generic-joses-[2019.05.25-16.55.59].vhdr',
   #  '../../../data/subject3/subject3-2/record-bv-generic-joses-[2019.05.25-17.10.22].vhdr',
   # '../../../data/subject4/subject4-1/record-bv-generic-juan-[2019.05.25-15.25.17].vhdr',
   # '../../../data/subject4/subject4-2/record-bv-generic-juan-[2019.05.25-15.45.01].vhdr',
   # '../../../data/subject5/subject5-1/record-bv-generic-manuel-[2019.06.15-17.34.06].vhdr',
   # '../../../data/subject5/subject5-2/record-bv-generic-manuel-[2019.06.15-17.46.45].vhdr',
   # '../../../data/subject6/subject6-1/record-bv-generic-marta-[2019.06.15-19.34.00].vhdr',
   # '../../../data/subject7/subject7-1/record-bv-generic-nati-[2019.04.27-19.11.05].vhdr',
   # '../../../data/subject8/subject8-1/record-bv-generic-santiago-[2019.06.15-18.11.29].vhdr',
   # '../../../data/subject8/subject8-2/record-bv-generic-santiago-[2019.06.15-18.24.23].vhdr'])

# Save a file with the maximum value of channel sum(abs(hits - no hits)) per experience in order
#utils.order("maximos.txt")

experiences_list = trainer.get_epochs_data_list([
                                                '../data/subject1/subject1-1/record-bv-generic-alex-[2019.06.15-18.43.41].vhdr',
                                                '../data/subject1/subject1-2/record-bv-generic-alex-[2019.06.15-19.04.43].vhdr'
                                                #'../data/subject2/subject2-1/record-bv-generic-gonza-[2019.05.25-14.37.56].vhdr',
                                                #'../data/subject2/subject2-2/record-bv-generic-gonza-[2019.05.25-14.43.22].vhdr'
                                                #'../data/subject3/subject3-1/record-bv-generic-joses-[2019.05.25-16.55.59].vhdr',
                                                #'../data/subject3/subject3-2/record-bv-generic-joses-[2019.05.25-17.10.22].vhdr'
                                                #'../data/subject4/subject4-1/record-bv-generic-juan-[2019.05.25-15.25.17].vhdr',
                                                #'../data/subject4/subject4-2/record-bv-generic-juan-[2019.05.25-15.45.01].vhdr'
                                                #'../data/subject5/subject5-1/record-bv-generic-manuel-[2019.06.15-17.34.06].vhdr',
                                                #'../data/subject5/subject5-2/record-bv-generic-manuel-[2019.06.15-17.46.45].vhdr'
                                                #'../data/subject6/subject6-1/record-bv-generic-marta-[2019.06.15-19.34.00].vhdr'
                                                #'../data/subject7/subject7-1/record-bv-generic-nati-[2019.04.27-19.11.05].vhdr',
                                                #'../data/subject8/subject8-1/record-bv-generic-santiago-[2019.06.15-18.11.29].vhdr',
                                                #'../data/subject8/subject8-2/record-bv-generic-santiago-[2019.06.15-18.24.23].vhdr'
                                                                  ])
train_experiences = experiences_list[:3]
test_experiences = experiences_list[3:]

classifier = trainer.train_classifier(train_experiences, "classifiers.joblib", n=10)
classification = tester.test_classifier("classifiers.joblib", test_experiences, n=10)
